from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except Exception:  # 不同版本路径可能挪窝，挪了就用相对目录兜底
    get_astrbot_data_path = None


PLUGIN_ID = "astrbot_plugin_sugar_debounce"
EXTRA_PREFIX = "sugar_debounce"

# 参与合并的组件类型。Reply 故意不收：引用链并进来会让上下文对不上
MERGEABLE_TYPES = {"Plain", "At", "Image", "Face", "Mface"}
TEXT_ONLY_TYPES = {"Plain", "At", "Face", "Mface"}

# 平台往 message_str 里塞的图片占位符
IMAGE_MARKERS = ("[图片]", "[Image]", "[image]", "[表情]", "[CQ:image")


@dataclass
class BufferedMessage:
    sequence: tuple[float, int]
    text: str
    components: list[Any] = field(default_factory=list)
    images: int = 0


@dataclass
class Burst:
    messages: list[BufferedMessage] = field(default_factory=list)
    current_event: AstrMessageEvent | None = None
    pipeline_task: asyncio.Task[Any] | None = None
    tools_started: bool = False
    published: bool = False
    updated_at: float = field(default_factory=time.monotonic)


@register(PLUGIN_ID, "sugarcane", "图文混发只回一次", "1.0.0")
class SugarDebouncePlugin(Star):
    """把没说完的话攒起来再交给模型，图片也跟着一起攒。"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config
        self._bursts: dict[str, Burst] = {}
        self._arrival_sequence = 0
        self._media_dir = self._resolve_media_dir()
        self._last_cleanup = 0.0

    # ---------- 钩子 ----------

    @filter.regex(r"^[\s\S]*$", priority=10000)
    async def stamp_message_arrival(self, event: AstrMessageEvent) -> None:
        """按到达顺序打序号，别让其它插件拖慢的先后顺序把话拼反"""
        if self._enabled():
            self._ensure_arrival_sequence(event)

    @filter.regex(r"^[\s\S]*$", priority=-10000)
    async def capture_incoming_message(self, event: AstrMessageEvent) -> None:
        """在默认流程启动前先扣下这条消息"""
        if self._is_early_llm_candidate(event):
            await self._capture(event)

    @filter.on_waiting_llm_request(priority=1000)
    async def register_pipeline_task(self, event: AstrMessageEvent) -> None:
        if event.get_extra(self._extra_key("captured"), False):
            key = event.get_extra(self._extra_key("key"), None)
        else:
            await self._capture(event)
            key = event.get_extra(self._extra_key("key"), None)

        if not isinstance(key, str):
            return
        burst = self._bursts.get(key)
        is_current = burst is not None and burst.current_event is event
        is_detached = bool(event.get_extra(self._extra_key("detached"), False))
        if not is_current and not is_detached:
            event.set_extra(self._extra_key("stale"), True)
            event.stop_event()
            return
        if not is_current or burst is None:
            return

        # 防抖窗口：首条是图就多等一会儿，让用户把配的字打出来
        delay_ms = self._delay_ms(burst)
        if delay_ms > 0:
            burst.published = True
            try:
                await asyncio.sleep(delay_ms / 1000.0)
            except asyncio.CancelledError:
                # 用户在窗口内又发了一条，_capture 会给这条 event 打 stale 再 cancel 掉它的
                # pipeline task —— 这是设计内的，不是故障。但 context_utils 用的是
                # `except BaseException`，取消信号会被记成一条 ERROR + Traceback。
                # 自己收掉，顺便 stop_event 让 pipeline 停在这里，别继续去调模型。
                # 插件重载、进程退出这类外部取消没有 stale 标记，照常往上抛。
                if event.get_extra(self._extra_key("stale"), False):
                    event.stop_event()
                    return
                raise
            fresh = self._bursts.get(key)
            if fresh is None or fresh.current_event is not event:
                event.set_extra(self._extra_key("stale"), True)
                event.stop_event()
                return
            burst = fresh

        burst.published = True
        self._apply_merged(event, burst)
        self._bind_pipeline_task(burst, key, event)

    @filter.on_llm_request(priority=1000)
    async def merge_request(self, event: AstrMessageEvent, request: Any) -> None:
        if event.get_extra(self._extra_key("stale"), False):
            event.stop_event()
            return

    @filter.on_using_llm_tool(priority=1000)
    async def mark_tool_started(
        self,
        event: AstrMessageEvent,
        tool: Any,
        tool_args: dict[str, Any] | None,
    ) -> None:
        key = event.get_extra(self._extra_key("key"), None)
        if not isinstance(key, str):
            return
        burst = self._bursts.get(key)
        if burst is not None and burst.current_event is event:
            burst.tools_started = True

    @filter.on_decorating_result(priority=1000)
    async def suppress_stale_result(self, event: AstrMessageEvent) -> None:
        if event.get_extra(self._extra_key("stale"), False):
            result = event.get_result()
            if result is not None and getattr(result, "chain", None) is not None:
                result.chain = []
            event.stop_event()

    # ---------- 收集 ----------

    async def _capture(self, event: AstrMessageEvent) -> None:
        if not self._enabled():
            return
        text = self._real_text(self._message_text(event))
        components = self._components(event)
        key = self._conversation_key(event)
        if not key or not self._mergeable(event, text, components):
            return

        image_count = sum(1 for c in components if type(c).__name__ == "Image")
        if not text and not image_count:
            return

        self._prune_bursts()
        self._cleanup_media()

        # 图片先落到自己目录：原消息被取消后它的临时文件随时会被清掉
        if image_count:
            components = await self._snapshot_images(components)

        max_messages = self._safe_int("max_messages", 8, minimum=0)
        max_chars = self._safe_int("max_chars", 4000, minimum=0)
        max_images = self._safe_int("max_images", 9, minimum=0)
        sequence = self._ensure_arrival_sequence(event)
        buffered = BufferedMessage(
            sequence=sequence,
            text=text,
            components=components,
            images=image_count,
        )

        burst = self._bursts.get(key)
        start_new = burst is None
        if burst is not None:
            start_new = (
                self._must_start_new_burst(burst)
                or self._would_exceed(burst, buffered, max_messages, max_chars, max_images)
            )
            if start_new:
                previous = burst.current_event
                if previous is not None:
                    previous.set_extra(self._extra_key("detached"), True)

        if start_new:
            burst = None
        if burst is None:
            burst = Burst()
            self._bursts[key] = burst
        else:
            previous = burst.current_event
            if previous is not None and previous is not event:
                previous.set_extra(self._extra_key("stale"), True)
                previous_task = burst.pipeline_task
                burst.pipeline_task = None
                if previous_task is not None and not previous_task.done():
                    previous_task.cancel()

        burst.messages.append(buffered)
        burst.messages.sort(key=lambda item: item.sequence)
        burst.current_event = event
        burst.published = False
        burst.updated_at = time.monotonic()

        event.set_extra(self._extra_key("captured"), True)
        event.set_extra(self._extra_key("key"), key)
        self._bind_pipeline_task(burst, key, event)

    def _apply_merged(self, event: AstrMessageEvent, burst: Burst) -> None:
        """把攒下的文本和组件链一起塞进当前 event"""
        texts = [m.text for m in burst.messages if m.text]
        merged_text = self._join_messages(texts)

        components: list[Any] = []
        for item in burst.messages:
            components.extend(item.components)

        if not merged_text and components:
            merged_text = str(self._config_get("image_only_prompt", "看看这个") or "").strip()
            merged_text = merged_text or "看看这个"

        event.message_str = merged_text
        message_obj = getattr(event, "message_obj", None)
        if message_obj is not None:
            message_obj.message_str = merged_text
            if components:
                message_obj.message = components

    # ---------- 判断 ----------

    def _is_early_llm_candidate(self, event: AstrMessageEvent) -> bool:
        if not self._enabled():
            return False
        stopped_getter = getattr(event, "is_stopped", None)
        if callable(stopped_getter) and stopped_getter():
            return False
        if bool(getattr(event, "_has_send_oper", False)):
            return False
        return bool(
            getattr(event, "call_llm", False)
            or getattr(event, "is_wake", False)
            or getattr(event, "is_at_or_wake_command", False)
        )

    def _must_start_new_burst(self, burst: Burst) -> bool:
        task = burst.pipeline_task
        if burst.tools_started or (task is not None and task.done()):
            return True
        previous = burst.current_event
        if previous is None:
            return False
        result_getter = getattr(previous, "get_result", None)
        result = result_getter() if callable(result_getter) else None
        content_type = getattr(result, "result_content_type", None)
        content_type_name = getattr(content_type, "name", str(content_type or ""))
        if "STREAMING" in content_type_name.upper():
            return True
        # 已经发出去回复了就别再攒，否则用户会看到"回一半又回一条"
        chain = getattr(result, "chain", None)
        return bool(chain)

    def _would_exceed(
        self,
        burst: Burst,
        message: BufferedMessage,
        max_messages: int,
        max_chars: int,
        max_images: int,
    ) -> bool:
        if max_messages > 0 and len(burst.messages) + 1 > max_messages:
            return True
        if max_chars > 0:
            current_chars = sum(len(item.text) for item in burst.messages)
            if current_chars + len(message.text) > max_chars:
                return True
        if max_images > 0:
            total_images = sum(item.images for item in burst.messages) + message.images
            if total_images > max_images:
                return True
        return False

    def _delay_ms(self, burst: Burst) -> int:
        if not burst.messages:
            return 0
        if burst.messages[0].images:
            return self._safe_int("image_debounce_ms", 1000, minimum=0)
        return self._safe_int("debounce_ms", 0, minimum=0)

    def _mergeable(
        self,
        event: AstrMessageEvent,
        text: str,
        components: list[Any],
    ) -> bool:
        message_obj = getattr(event, "message_obj", None)
        original_text = getattr(message_obj, "message_str", None)
        prefix_text = (
            original_text.strip()
            if isinstance(original_text, str) and original_text.strip()
            else text
        )

        if bool(self._config_get("ignore_prefixes_enabled", True)):
            prefixes = self._config_get("ignore_prefixes", ["/", "!"])
            if isinstance(prefixes, list) and any(
                prefix_text.startswith(item)
                for item in prefixes
                if isinstance(item, str) and item
            ):
                return False

        group_getter = getattr(event, "get_group_id", None)
        group_id = group_getter() if callable(group_getter) else None
        is_group_message = group_id is not None and bool(str(group_id).strip())
        if is_group_message and bool(self._config_get("required_prefixes_enabled", False)):
            required = self._config_get("required_prefixes", [])
            matched = isinstance(required, list) and any(
                prefix_text.startswith(item)
                for item in required
                if isinstance(item, str) and item
            )
            if not matched and not self._mentions_bot(event, components):
                return False

        if not isinstance(components, (list, tuple)):
            return True
        allowed = MERGEABLE_TYPES if self._merge_images() else TEXT_ONLY_TYPES
        return all(type(component).__name__ in allowed for component in components)

    def _mentions_bot(self, event: AstrMessageEvent, components: Any) -> bool:
        if not isinstance(components, (list, tuple)):
            return False
        self_id_getter = getattr(event, "get_self_id", None)
        self_id = self_id_getter() if callable(self_id_getter) else None
        if self_id is None or not str(self_id).strip():
            return False
        return any(
            type(component).__name__ == "At"
            and str(getattr(component, "qq", "")) == str(self_id)
            for component in components
        )

    # ---------- 图片 ----------

    async def _snapshot_images(self, components: list[Any]) -> list[Any]:
        out: list[Any] = []
        for component in components:
            if type(component).__name__ == "Image":
                out.append(await self._snapshot_image(component))
            else:
                out.append(component)
        return out

    async def _snapshot_image(self, component: Any) -> Any:
        try:
            path = await component.convert_to_file_path()
        except Exception as exc:
            logger.debug(f"[{PLUGIN_ID}] 图片落地失败，保留原组件：{exc!r}")
            return component
        try:
            cached = await asyncio.to_thread(self._cache_image, path)
        except Exception as exc:
            logger.debug(f"[{PLUGIN_ID}] 图片缓存失败，直接用原路径：{exc!r}")
            cached = path
        try:
            return Image(file=Path(cached).as_uri(), path=str(cached))
        except Exception:
            return component

    def _cache_image(self, path: str) -> str:
        source = Path(path)
        if not source.is_file():
            return path
        digest = hashlib.sha1()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        suffix = source.suffix if source.suffix else ".jpg"
        target = self._media_dir / f"{digest.hexdigest()[:20]}{suffix}"
        if not target.exists():
            shutil.copy2(source, target)
        return str(target)

    def _resolve_media_dir(self) -> Path:
        root: Path | None = None
        try:
            data_path = get_astrbot_data_path() if get_astrbot_data_path else None
            if data_path:
                root = Path(str(data_path)) / "plugin_data" / PLUGIN_ID / "media"
        except Exception:
            root = None
        if root is None:
            root = Path(__file__).resolve().parent / "data_fallback" / "media"
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(f"[{PLUGIN_ID}] 图片缓存目录不可用：{exc!r}")
        return root

    def _cleanup_media(self, max_age: float = 900.0) -> None:
        now = time.monotonic()
        if now - self._last_cleanup < 60.0:
            return
        self._last_cleanup = now
        cutoff = time.time() - max_age
        try:
            for item in self._media_dir.iterdir():
                try:
                    if item.is_file() and item.stat().st_mtime < cutoff:
                        item.unlink()
                except OSError:
                    continue
        except Exception:
            pass

    # ---------- 杂项 ----------

    def _components(self, event: AstrMessageEvent) -> list[Any]:
        message_obj = getattr(event, "message_obj", None)
        components = getattr(message_obj, "message", None)
        if isinstance(components, (list, tuple)):
            return list(components)
        return []

    def _message_text(self, event: AstrMessageEvent) -> str:
        text = getattr(event, "message_str", "")
        return text.strip() if isinstance(text, str) else ""

    def _real_text(self, text: str) -> str:
        out = text
        for marker in IMAGE_MARKERS:
            out = out.replace(marker, "")
        return out.strip()

    def _conversation_key(self, event: AstrMessageEvent) -> str | None:
        sender_id = self._sender_id(event)
        if not sender_id:
            return None
        group_getter = getattr(event, "get_group_id", None)
        group_id = group_getter() if callable(group_getter) else None
        scope = f"group:{group_id}" if group_id is not None else "private"
        return f"{event.unified_msg_origin}|{scope}|user:{sender_id}"

    def _sender_id(self, event: AstrMessageEvent) -> str | None:
        getter = getattr(event, "get_sender_id", None)
        if callable(getter):
            value = getter()
            if value is not None and str(value).strip():
                return str(value).strip()
        message_obj = getattr(event, "message_obj", None)
        sender = getattr(message_obj, "sender", None)
        for name in ("user_id", "uin", "sender_id", "id"):
            value = getattr(sender, name, None)
            if value is not None and str(value).strip():
                return str(value).strip()
        raw = getattr(message_obj, "raw_message", None)
        if isinstance(raw, dict):
            value = raw.get("user_id") or raw.get("sender_id")
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    def _prune_bursts(self) -> None:
        cutoff = time.monotonic() - 120.0
        expired = [
            key
            for key, burst in self._bursts.items()
            if burst.updated_at < cutoff
            and (burst.pipeline_task is None or burst.pipeline_task.done())
        ]
        for key in expired:
            self._bursts.pop(key, None)

    def _on_pipeline_done(
        self,
        key: str,
        event: AstrMessageEvent,
        task: asyncio.Task[Any],
    ) -> None:
        burst = self._bursts.get(key)
        if (
            burst is not None
            and burst.current_event is event
            and burst.pipeline_task is task
        ):
            self._bursts.pop(key, None)

    def _bind_pipeline_task(
        self,
        burst: Burst,
        key: str,
        event: AstrMessageEvent,
    ) -> None:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is None or burst.pipeline_task is task:
            return
        burst.pipeline_task = task
        task.add_done_callback(
            lambda completed, burst_key=key, current_event=event: self._on_pipeline_done(
                burst_key,
                current_event,
                completed,
            )
        )

    def _join_messages(self, messages: list[str]) -> str:
        return "\n".join(messages)

    def _ensure_arrival_sequence(
        self,
        event: AstrMessageEvent,
    ) -> tuple[float, int]:
        key = self._extra_key("arrival_sequence")
        existing = event.get_extra(key, None)
        if (
            isinstance(existing, tuple)
            and len(existing) == 2
            and isinstance(existing[0], int | float)
            and isinstance(existing[1], int)
        ):
            return float(existing[0]), existing[1]

        self._arrival_sequence += 1
        try:
            created_at = float(getattr(event, "created_at", time.time()))
        except (TypeError, ValueError):
            created_at = time.time()
        sequence = (created_at, self._arrival_sequence)
        event.set_extra(key, sequence)
        return sequence

    def _enabled(self) -> bool:
        return bool(self._config_get("enabled", True))

    def _merge_images(self) -> bool:
        return bool(self._config_get("merge_images", True))

    def _safe_int(self, key: str, default: int, minimum: int) -> int:
        try:
            return max(minimum, int(self._config_get(key, default)))
        except (TypeError, ValueError):
            logger.warning(f"[{PLUGIN_ID}] 配置 {key} 不是整数，改用 {default}")
            return default

    def _config_get(self, key: str, default: Any) -> Any:
        if self.config is None:
            return default
        getter = getattr(self.config, "get", None)
        return getter(key, default) if callable(getter) else getattr(self.config, key, default)

    def _extra_key(self, name: str) -> str:
        return f"{EXTRA_PREFIX}.{name}"
