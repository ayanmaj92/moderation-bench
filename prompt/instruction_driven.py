import os
from .base_prompt import BasePrompt

_POLICIES_DIR = os.path.join(os.path.dirname(__file__), "policies")


class InstructionDrivenPrompt(BasePrompt):
    def __init__(self, mode, platform="bluesky", output_format="output_bluesky"):
        self.mode = mode
        self.platform = platform
        self.system_text = self._load(_POLICIES_DIR, "systemprompt.md")
        self.policy_text = self._load(_POLICIES_DIR, platform, f"{mode}.md")
        self.output_text = self._load(_POLICIES_DIR, f"{output_format}.md")

    def _load(self, *path_parts):
        with open(os.path.join(*path_parts), "r") as f:
            return f.read().strip()

    def _append_query_block(self, user_content, query_text, query_images,
                            query_video, video_as_frames):
        """Append the query post's media blocks, its text/media summary, and the
        output-format instructions to ``user_content``. Shared verbatim by the
        instruction-driven and example-driven prompt builders -- keep it here so the two cannot
        drift."""
        media_lines = []

        if query_images is not None:
            valid_images = [(i + 1, img) for i, img in enumerate(query_images) if img is not None]
            for _, img in valid_images:
                user_content.append({"type": "image_url", "image_url": {"url": img}})
            media_lines = [f" - Image {i}: <image>" for i, _ in valid_images]
        elif query_video is not None:
            if not video_as_frames:
                user_content.append({"type": "video_url", "video_url": {"url": query_video}})
                media_lines = [" - Video: <video>"]
            else:
                valid_frames = [(i + 1, frame) for i, frame in enumerate(query_video) if frame is not None]
                for _, frame in valid_frames:
                    user_content.append({"type": "image_url", "image_url": {"url": frame}})
                media_lines = [f" - Video Frame {i}: <image>" for i, _ in valid_frames]

        if not query_text:
            query_text_ref = "<no text was posted.>"
        else:
            query_text_ref = query_text.replace("\n", "\\n")

        if not media_lines:
            media_lines = [" - <no visual content was posted.>"]

        post_block = f" - Text: {query_text_ref}\n" + "\n".join(media_lines)
        user_content.append({
            "type": "text",
            "text": f"{post_block}\n\n--------------------\n\n{self.output_text}",
        })
        return user_content

    def build_chat_messages(self, **kwargs):
        query_text = kwargs.get("query_text")
        query_images = kwargs.get("query_images")
        query_video = kwargs.get("query_video")
        video_as_frames = kwargs.get("video_as_frames", False)

        messages = [{"role": "system", "content": self.system_text}]
        user_content = [
            {"type": "text", "text": f"{self.policy_text}\n# *SOCIAL MEDIA POST TO BE CATEGORIZED*\n"}
        ]
        self._append_query_block(user_content, query_text, query_images, query_video, video_as_frames)

        messages.append({"role": "user", "content": user_content})
        return messages
