#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-10-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
import cv2


def max_resize(img, max_width=1024, max_height=None):
    assert (max_width is None) != (max_height is None)

    if max_width is not None and img.shape[1] > max_width:
        s = max_width / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=s, fy=s)

    elif max_height is not None and img.shape[0] > max_height:
        s = max_height / img.shape[0]
        img = cv2.resize(img, (0, 0), fx=s, fy=s)

    return img
