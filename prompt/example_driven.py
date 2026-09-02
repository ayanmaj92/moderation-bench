from .instruction_driven import InstructionDrivenPrompt


class ExampleDrivenPrompt(InstructionDrivenPrompt):
    def __init__(self, mode, platform="bluesky", output_format="output_bluesky", **kwargs):
        super().__init__(mode, platform=platform, output_format=output_format)
        self.use_safe_examples = kwargs.get("use_safe_examples", True)

    def build_chat_messages(self, **kwargs):
        query_text = kwargs.get("query_text")
        query_images = kwargs.get("query_images")
        query_video = kwargs.get("query_video")
        video_as_frames = kwargs.get("video_as_frames", False)

        example_k_texts = kwargs.get("example_k_texts", [])
        example_k_images = kwargs.get("example_k_images", [])
        example_k_labels = kwargs.get("example_k_labels", [])
        assert len(example_k_texts) == len(example_k_images) == len(example_k_labels), \
            "Example-driven examples lists must be of the same length."

        messages = [{"role": "system", "content": self.system_text}]
        user_content = [{"type": "text", "text": f"{self.policy_text}\n"}]

        ##### FEW SHOT SETUP #####
        num_examples = len(example_k_labels)
        if num_examples > 0:
            user_content.append({"type": "text", "text": (
                "Below are example Social Media Posts with texts and/or images and their "
                "corresponding moderation labels.\n--------------------\n# EXAMPLE SOCIAL MEDIA POSTS"
            )})

        shown = 0
        for i in range(num_examples):
            if not self.use_safe_examples and example_k_labels[i] in ("S0", "safe"):
                continue
            shown += 1
            post_num = shown
            text = example_k_texts[i] if example_k_texts[i] else "<no text was posted.>"
            label = example_k_labels[i]
            img_item = example_k_images[i]

            if img_item:
                valid_images = [(idx + 1, img) for idx, img in enumerate(img_item) if img is not None]
                for _, img in valid_images:
                    user_content.append({"type": "image_url", "image_url": {"url": img}})
                img_txt = "\n".join(f"Image {idx}: <image>" for idx, _ in valid_images)
            else:
                img_txt = "<no image was posted for sample.>"

            safe_text = text.replace("\n", "\\n")
            user_content.append({
                "type": "text",
                "text": f"\nPost Example {post_num}:\nText: {safe_text}\nImages: {img_txt}\nLabel: {label}"
            })

        if not self.use_safe_examples:
            reminder_text = (
                "\n**Remember**: These examples only illustrate unsafe categories, but you must "
                "still predict 'S0' if you judge the input Query Post given next is SAFE.\n"
            )
        else:
            reminder_text = ""
        if shown > 0:
            user_content[-1]["text"] += f"\n{reminder_text}\n"

        user_content.append(
            {"type": "text", "text": "\n--------------------\n# *SOCIAL MEDIA POST TO BE CATEGORIZED*\n"}
        )

        ##### QUERY SETUP #####
        self._append_query_block(user_content, query_text, query_images, query_video, video_as_frames)

        messages.append({"role": "user", "content": user_content})
        return messages
