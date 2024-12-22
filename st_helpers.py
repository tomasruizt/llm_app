def is_video(media_file) -> bool:
    return media_file.name.endswith(".mp4")


def is_image(media_file) -> bool:
    return media_file.name.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
