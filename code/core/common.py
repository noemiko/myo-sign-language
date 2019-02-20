import struct
from struct import error


def pack(fmt, *args):
    return struct.pack('<' + fmt, *args)


def unpack(fmt, *args):
    try:
        return struct.unpack('<' + fmt, *args)
    except error as er:
        print(er)
        return (0, 0, 0)


def text(scr, font, txt, pos, clr=(255, 255, 255)):
    scr.blit(font.render(txt, True, clr), pos)
