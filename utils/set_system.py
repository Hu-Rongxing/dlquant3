import win32api
import win32gui
from win32con import WM_INPUTLANGCHANGEREQUEST


def change_language(language="EN"):
    """
    切换语言
    :param language: EN––English; ZH––Chinese
    :return: bool
    """
    LANGUAGE = {
        "CH": 0x0804,
        "EN": 0x0409
    }
    """
    获取键盘布局
    im_list = win32api.GetKeyboardLayoutList()
    im_list = list(map(hex, im_list))
    print(im_list)
    """
    hwnd = win32gui.GetForegroundWindow()
    language = LANGUAGE.get(language)
    result = win32api.SendMessage(
        hwnd,
        WM_INPUTLANGCHANGEREQUEST,
        0,
        language
    )
    return result == 0

if __name__ == '__main__':
    change_language()