"""
_svg.py
=======

Make matplotlib SVGs easy to edit in Illustrator and PowerPoint.

save_svg(fig, path) writes the figure and then
  * removes every clip path (Illustrator turns them into clipping masks),
  * replaces every <use> reference by an inline copy of the referenced path
    (matplotlib reuses marker shapes; PowerPoint's "Convert to Shape" handles copies better),
  * leaves text as text (requires rcParams["svg.fonttype"] = "none"), so labels stay editable,
  * omits the time stamp, so identical runs write identical files.
Group ids set with artist.set_gid(...) survive and appear as layer / group names.

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import copy
import re
import xml.etree.ElementTree as ET

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)


def _q(tag):
    return f"{{{SVG}}}{tag}"


def flatten(path):
    tree = ET.parse(path)
    root = tree.getroot()
    parent = {c: p for p in root.iter() for c in p}

    # 1) drop clip paths
    for el in root.iter():
        el.attrib.pop("clip-path", None)
        if "style" in el.attrib and "clip-path" in el.attrib["style"]:
            el.attrib["style"] = re.sub(r"clip-path:[^;]*;?", "", el.attrib["style"])
    for cp in list(root.iter(_q("clipPath"))):
        parent[cp].remove(cp)

    # 2) inline <use> references
    by_id = {el.attrib["id"]: el for el in root.iter() if "id" in el.attrib}
    parent = {c: p for p in root.iter() for c in p}
    n_use = 0
    for use in list(root.iter(_q("use"))):
        href = use.attrib.get(f"{{{XLINK}}}href") or use.attrib.get("href")
        if not href or href[1:] not in by_id:
            continue
        src = copy.deepcopy(by_id[href[1:]])
        src.attrib.pop("id", None)
        x, y = use.attrib.get("x", "0"), use.attrib.get("y", "0")
        transform = f"translate({x} {y})"
        if "transform" in use.attrib:
            transform = use.attrib["transform"] + " " + transform
        if "transform" in src.attrib:
            transform = transform + " " + src.attrib["transform"]
        src.attrib["transform"] = transform
        style = "; ".join(s for s in (src.attrib.get("style", ""), use.attrib.get("style", "")) if s)
        if style:
            src.attrib["style"] = style
        p = parent[use]
        idx = list(p).index(use)
        p.remove(use)
        p.insert(idx, src)
        n_use += 1

    # 3) remove definitions that are no longer referenced
    text = ET.tostring(root, encoding="unicode")
    for d in list(root.iter(_q("defs"))):
        for child in list(d):
            cid = child.attrib.get("id")
            if cid and text.count(f"#{cid}") == 0 and child.tag != _q("style"):
                d.remove(child)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return n_use


def save_svg(fig, path, **kw):
    kw.setdefault("metadata", {"Date": None})    # no time stamp: identical runs give identical files
    fig.savefig(path, format="svg", **kw)
    flatten(path)
