# docx_track.py
"""Edit the mSystems revision Word documents in place.

TrackedDocx makes tracked changes (w:ins / w:del) in a manuscript .docx. Edits
address the accepted view of a paragraph: text inside existing insertions
counts, deleted text does not. Replacing text that lies inside the editing
author's own existing insertion edits that insertion directly (as Word does);
other text is marked as a tracked deletion followed by a tracked insertion.
Runs inside hyperlinks (Paperpile citations), fields, content controls and
smart tags are never edited; such an edit raises instead.

LetterDocx edits the response letter without tracking, cloning the formatting
of an existing answered comment (bold "Response:", italic labels, indented
quotes).

All other parts of the .docx package are copied byte-for-byte.
"""

import copy
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"w": W, "r": R, "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
      "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture"}
REL_IMAGE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
PKG_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
W14_IDS = ("{http://schemas.microsoft.com/office/word/2010/wordml}paraId",
           "{http://schemas.microsoft.com/office/word/2010/wordml}textId")
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
PROTECTED = {"hyperlink", "fldSimple", "sdt", "smartTag"}
ORIGINAL_GREEN, MODIFIED_BLUE = "38761d", "0000ff"  # response letter: Original quotes green, Modified blue
COLOR_FOLLOWERS = {"spacing", "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd",
                   "fitText", "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath"}


def q(tag: str) -> str:
    return f"{{{W}}}{tag}"


def local(el) -> str:
    return etree.QName(el).localname


class _Package:
    """A .docx zip whose word/document.xml is parsed and rewritten on save."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as z:
            self._infos = z.infolist()
            self._parts = {i.filename: z.read(i.filename) for i in self._infos}
        self.root = etree.fromstring(self._parts["word/document.xml"])

    def paragraphs(self) -> list:
        return list(self.root.iter(q("p")))

    @staticmethod
    def is_deleted(el) -> bool:
        return any(local(a) == "del" for a in el.iterancestors())

    def _normalize(self, p):
        """Split visible runs holding several w:t elements into one run per content
        element (same formatting), so each run carries at most one w:t."""
        for r in list(p.iter(q("r"))):
            if self.is_deleted(r) or len(r.findall(q("t"))) < 2:
                continue
            rpr = r.find(q("rPr"))
            for child in [c for c in r if c.tag != q("rPr")]:
                part = etree.Element(q("r"))
                if rpr is not None:
                    part.append(copy.deepcopy(rpr))
                part.append(child)
                r.addprevious(part)
            r.getparent().remove(r)

    def text_runs(self, p) -> list:
        """Visible runs with text, in document order: [(run, w:t element)]."""
        self._normalize(p)
        return [(r, r.find(q("t"))) for r in p.iter(q("r"))
                if not self.is_deleted(r) and r.find(q("t")) is not None]

    def accepted_text(self, p) -> str:
        return "".join(t.text or "" for t in p.iter(q("t")) if not self.is_deleted(t))

    def rejected_text(self, p) -> str:
        """Paragraph text with all tracked changes rejected (the original wording)."""
        out = []
        for el in p.iter(q("t"), q("delText")):
            inserted = any(local(a) == "ins" for a in el.iterancestors())
            if el.tag == q("delText") or not inserted:
                out.append(el.text or "")
        return "".join(out)

    def find(self, snippet: str, exact: bool = False):
        """The unique paragraph containing `snippet` (or, if `exact`, whose whole text it is)."""
        hits = [p for p in self.paragraphs()
                if (snippet == self.accepted_text(p).strip() if exact else snippet in self.accepted_text(p))]
        if len(hits) != 1:
            raise ValueError(f"{len(hits)} paragraphs contain {snippet[:70]!r}")
        return hits[0]

    def read_part(self, name: str) -> bytes:
        return self._parts[name]

    def replace_part(self, name: str, data: bytes):
        """Replace a package part (e.g. word/media/image1.png); not a tracked change."""
        if name not in self._parts:
            raise KeyError(name)
        self._parts[name] = data

    def add_part(self, name: str, data: bytes):
        """Add a new package part (e.g. word/media/image6.png)."""
        if name in self._parts:
            raise KeyError(f"{name} already exists")
        info = zipfile.ZipInfo(name, date_time=self._infos[0].date_time)
        info.compress_type = zipfile.ZIP_DEFLATED
        self._infos.append(info)
        self._parts[name] = data

    def _rels(self):
        return etree.fromstring(self._parts["word/_rels/document.xml.rels"])

    def add_image(self, media_name: str, png: bytes) -> str:
        """Add word/media/<media_name> and an image relationship; returns its rId."""
        self.add_part(f"word/media/{media_name}", png)
        rels = self._rels()
        n = max(int(r.get("Id")[3:]) for r in rels if r.get("Id", "").startswith("rId")) + 1
        rel = etree.SubElement(rels, f"{{{PKG_RELS}}}Relationship")
        rel.set("Id", f"rId{n}")
        rel.set("Type", REL_IMAGE)
        rel.set("Target", f"media/{media_name}")
        self._parts["word/_rels/document.xml.rels"] = etree.tostring(
            rels, xml_declaration=True, encoding="UTF-8", standalone=True)
        return f"rId{n}"

    def image_rid(self, media_name: str) -> str:
        hits = [r.get("Id") for r in self._rels() if r.get("Target") == f"media/{media_name}"]
        if len(hits) != 1:
            raise ValueError(f"{len(hits)} relationships target media/{media_name}")
        return hits[0]

    def set_image_extent(self, media_name: str, cx: int, cy: int):
        """Resize the frame of the (single) drawing showing media/<media_name>; not tracked."""
        blips = self.root.xpath(f"//a:blip[@r:embed='{self.image_rid(media_name)}']", namespaces=NS)
        if len(blips) != 1:
            raise ValueError(f"{len(blips)} drawings show {media_name}")
        inline = next(a for a in blips[0].iterancestors() if local(a) in ("inline", "anchor"))
        for ext in [inline.find("wp:extent", NS)] + inline.xpath(".//pic:spPr/a:xfrm/a:ext", namespaces=NS):
            ext.set("cx", str(cx))
            ext.set("cy", str(cy))

    def backup(self, backup_dir: Path) -> Path:
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = backup_dir / f"{self.path.stem}.before-{stamp}{self.path.suffix}"
        shutil.copy2(self.path, dest)
        return dest

    def save(self, out_path: Path | None = None):
        out_path = Path(out_path or self.path)
        self._parts["word/document.xml"] = etree.tostring(
            self.root, xml_declaration=True, encoding="UTF-8", standalone=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for info in self._infos:
                z.writestr(info, self._parts[info.filename])
        tmp.replace(out_path)


class TrackedDocx(_Package):
    def __init__(self, path: Path, author: str, date: str | None = None):
        super().__init__(path)
        self.author = author
        self.date = date or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ids = [int(v) for el in self.root.iter() for k, v in el.attrib.items()
               if k == q("id") and v.lstrip("-").isdigit()]
        self._next_id = max(ids, default=0) + 1
        self.log = []

    # -- helpers ---------------------------------------------------------------
    def _rev(self, kind: str):
        el = etree.Element(q(kind))
        el.set(q("id"), str(self._next_id))
        el.set(q("author"), self.author)
        el.set(q("date"), self.date)
        self._next_id += 1
        return el

    def _own_ins(self, el) -> bool:
        return local(el) == "ins" and el.get(q("author")) == self.author

    def _check_editable(self, run):
        for a in run.iterancestors():
            if local(a) in PROTECTED:
                raise ValueError(f"refusing to edit text inside w:{local(a)}")
            if local(a) == "ins" and not self._own_ins(a):
                raise ValueError("refusing to edit another author's tracked insertion")
            if local(a) == "p":
                break

    @staticmethod
    def _split(run, t, k: int):
        """Split run at character k of its text; returns the new right-hand run."""
        right = copy.deepcopy(run)
        rt = right.find(q("t"))
        text = t.text or ""
        t.text, rt.text = text[:k], text[k:]
        t.set(XML_SPACE, "preserve")
        rt.set(XML_SPACE, "preserve")
        run.addnext(right)
        return right

    def _isolate(self, p, start: int, end: int) -> list:
        """Split runs so [start, end) is exactly covered by whole runs; return them."""
        for boundary in (start, end):
            pos = 0
            for r, t in self.text_runs(p):
                n = len(t.text or "")
                if pos < boundary < pos + n:
                    self._split(r, t, boundary - pos)
                    break
                pos += n
        covered, pos = [], 0
        for r, t in self.text_runs(p):
            n = len(t.text or "")
            if start <= pos and pos + n <= end and n:
                covered.append(r)
            pos += n
        return covered

    def _new_run(self, template_run, text: str):
        run = etree.Element(q("r"))
        rpr = template_run.find(q("rPr"))
        if rpr is not None:
            rpr = copy.deepcopy(rpr)
            for change in rpr.findall(q("rPrChange")):
                rpr.remove(change)
            run.append(rpr)
        t = etree.SubElement(run, q("t"))
        t.text = text
        t.set(XML_SPACE, "preserve")
        return run

    # -- edits -----------------------------------------------------------------
    def replace(self, old: str, new: str, context: str | None = None):
        """Tracked replacement of `old` by `new` in the paragraph containing `context` (or `old`)."""
        p = self.find(context or old)
        text = self.accepted_text(p)
        if text.count(old) != 1:
            raise ValueError(f"{text.count(old)} occurrences of {old[:60]!r} in the paragraph")
        start = text.index(old)
        covered = self._isolate(p, start, start + len(old))
        for r in covered:
            self._check_editable(r)

        last = covered[-1]
        insertion = None
        if new:
            new_run = self._new_run(covered[0], new)
            if self._own_ins(last.getparent()):
                last.addnext(new_run)
            else:
                insertion = self._rev("ins")
                insertion.append(new_run)

        for r in covered:
            parent = r.getparent()
            if self._own_ins(parent):
                parent.remove(r)
                if not parent.findall(q("r")):
                    parent.getparent().remove(parent)
            else:
                deletion = self._rev("del")
                r.addprevious(deletion)
                deletion.append(r)
                t = r.find(q("t"))
                t.tag = q("delText")
                t.set(XML_SPACE, "preserve")
        if insertion is not None:
            last.getparent().addnext(insertion)  # after the w:del wrapping the last run
        self.log.append({"action": "replace", "old": old, "new": new})

    def _mark_inserted(self, p):
        """Mark paragraph p as a tracked insertion: its runs and its paragraph mark."""
        for a in W14_IDS:
            p.attrib.pop(a, None)
        ppr = p.find(q("pPr"))
        if ppr is None:
            ppr = etree.Element(q("pPr"))
            p.insert(0, ppr)
        rpr = ppr.find(q("rPr"))
        if rpr is None:
            rpr = etree.SubElement(ppr, q("rPr"))
        for old in rpr.findall(q("ins")):
            rpr.remove(old)
        rpr.insert(0, self._rev("ins"))
        runs = [r for r in p if r.tag == q("r")]
        if runs:
            insertion = self._rev("ins")
            runs[0].addprevious(insertion)
            for r in runs:
                insertion.append(r)

    def insert_paragraph_after(self, after, runs: list[tuple], ppr_from=None):
        """Tracked new paragraph after paragraph `after`. `runs` is [(rPr element or None, text)];
        paragraph properties are copied from `ppr_from` (default: `after`). Returns the paragraph."""
        p = etree.Element(q("p"))
        src = (ppr_from if ppr_from is not None else after).find(q("pPr"))
        if src is not None:
            p.append(copy.deepcopy(src))
        for rpr, text in runs:
            r = etree.SubElement(p, q("r"))
            if rpr is not None:
                rpr = copy.deepcopy(rpr)
                for change in rpr.findall(q("rPrChange")):
                    rpr.remove(change)
                r.append(rpr)
            t = etree.SubElement(r, q("t"))
            t.text = text
            t.set(XML_SPACE, "preserve")
        self._mark_inserted(p)
        after.addnext(p)
        self.log.append({"action": "insert_paragraph", "text": "".join(t for _, t in runs)})
        return p

    def insert_image_after(self, after, template, media_name: str, png: bytes, width_emu: int | None = None):
        """Tracked new paragraph after `after` showing a new image, cloned from the drawing
        paragraph `template` (same alignment and frame width unless `width_emu`). Returns it."""
        from PIL import Image
        import io
        p = copy.deepcopy(template)
        for r in p.findall(q("r")):
            if r.find(q("drawing")) is None:
                p.remove(r)
        rid = self.add_image(media_name, png)
        p.xpath(".//a:blip", namespaces=NS)[0].set(f"{{{R}}}embed", rid)
        inline = p.xpath(".//wp:inline | .//wp:anchor", namespaces=NS)[0]
        img = Image.open(io.BytesIO(png))
        cx = width_emu or int(inline.find("wp:extent", NS).get("cx"))
        cy = int(round(cx * img.height / img.width))
        for ext in [inline.find("wp:extent", NS)] + inline.xpath(".//pic:spPr/a:xfrm/a:ext", namespaces=NS):
            ext.set("cx", str(cx))
            ext.set("cy", str(cy))
        ids = [int(d.get("id")) for d in self.root.xpath("//wp:docPr", namespaces=NS)]
        doc_pr = inline.find("wp:docPr", NS)
        doc_pr.set("id", str(max(ids) + 1))
        doc_pr.set("name", media_name)
        p.xpath(".//pic:cNvPr", namespaces=NS)[0].set("name", media_name)
        self._mark_inserted(p)
        after.addnext(p)
        self.log.append({"action": "insert_image", "media": media_name})
        return p

    def insert_after(self, anchor: str, new: str, context: str | None = None):
        """Tracked insertion of `new` immediately after the text `anchor`."""
        p = self.find(context or anchor)
        text = self.accepted_text(p)
        if text.count(anchor) != 1:
            raise ValueError(f"{text.count(anchor)} occurrences of {anchor[:60]!r} in the paragraph")
        end = text.index(anchor) + len(anchor)
        last = self._isolate(p, end - len(anchor), end)[-1]
        self._check_editable(last)
        new_run = self._new_run(last, new)
        if self._own_ins(last.getparent()):
            last.addnext(new_run)
        else:
            insertion = self._rev("ins")
            insertion.append(new_run)
            last.addnext(insertion)
        self.log.append({"action": "insert_after", "anchor": anchor, "new": new})


class LetterDocx(_Package):
    """Untracked edits to the response letter, in the style of an answered comment."""

    def __init__(self, path: Path, template_response: str):
        """`template_response`: text unique to an answered Response paragraph that is
        followed by an 'Original (...)' label and an indented quote."""
        super().__init__(path)
        resp = self.find(template_response)
        label = resp.getnext()
        if not self.accepted_text(label).startswith("Original"):
            raise ValueError("template response is not followed by an 'Original' label")
        quote = label.getnext()
        self._t_resp, self._t_label, self._t_quote = resp, label, quote
        runs = resp.findall(q("r"))
        self._bold_rpr = copy.deepcopy(runs[0].find(q("rPr")))
        self._plain_rpr = copy.deepcopy(runs[-1].find(q("rPr")))

    @staticmethod
    def _run(rpr, text: str):
        r = etree.Element(q("r"))
        if rpr is not None:
            r.append(copy.deepcopy(rpr))
        t = etree.SubElement(r, q("t"))
        t.text = text
        t.set(XML_SPACE, "preserve")
        return r

    def _clone_paragraph(self, template, text: str):
        p = etree.Element(q("p"))
        ppr = template.find(q("pPr"))
        if ppr is not None:
            p.append(copy.deepcopy(ppr))
        rpr = template.find(q("r")).find(q("rPr"))
        p.append(self._run(rpr, text))
        return p

    def response_paragraph_after(self, comment_start: str):
        """The 'Response:' paragraph following the comment that starts with `comment_start`."""
        comment = next(p for p in self.paragraphs() if self.accepted_text(p).startswith(comment_start))
        p = comment.getnext()
        while p is not None and not self.accepted_text(p).startswith("Response:"):
            p = p.getnext()
        if p is None:
            raise ValueError(f"no Response paragraph after {comment_start!r}")
        return p

    def set_response(self, comment_start: str, text: str):
        p = self.response_paragraph_after(comment_start)
        for r in p.findall(q("r")):
            p.remove(r)
        p.append(self._run(self._bold_rpr, "Response:"))
        p.append(self._run(self._plain_rpr, " " + text))
        return p

    def set_text(self, snippet: str, text: str, exact: bool = False):
        """Replace the whole text of the paragraph containing `snippet` (or, if `exact`,
        consisting of it), keeping the formatting of its first run (used for quote paragraphs)."""
        return self.set_paragraph_text(self.find(snippet, exact), text)

    def set_paragraph_text(self, p, text: str):
        """Replace the whole text of paragraph p, keeping the formatting of its first run."""
        runs = p.findall(q("r"))
        rpr = runs[0].find(q("rPr"))
        for r in runs:
            p.remove(r)
        p.append(self._run(rpr, text))
        return p

    def replace_in_run(self, old: str, new: str):
        """Untracked replacement inside the single text element that contains `old`,
        leaving the run's formatting (e.g. highlighting) as it is."""
        hits = [t for t in self.root.iter(q("t")) if old in (t.text or "")]
        if len(hits) != 1:
            raise ValueError(f"{len(hits)} text elements contain {old[:60]!r}")
        hits[0].text = hits[0].text.replace(old, new)

    def response_text(self, comment_start: str) -> str:
        text = self.accepted_text(self.response_paragraph_after(comment_start))
        return text[len("Response:"):].strip()

    @staticmethod
    def _set_color(p, color: str):
        """Set the text colour of every run (and the paragraph mark) of p."""
        rprs = [r.find(q("rPr")) for r in p.iter(q("r"))]
        ppr = p.find(q("pPr"))
        if ppr is not None and ppr.find(q("rPr")) is not None:
            rprs.append(ppr.find(q("rPr")))
        for rpr in rprs:
            if rpr is None:
                continue
            c = rpr.find(q("color"))
            if c is None:  # w:color precedes these run properties in the schema
                c = etree.Element(q("color"))
                later = [x for x in rpr if local(x) in COLOR_FOLLOWERS]
                later[0].addprevious(c) if later else rpr.append(c)
            for a in list(c.attrib):
                if etree.QName(a).localname.startswith("theme"):
                    del c.attrib[a]
            c.set(q("val"), color)

    def add_pairs(self, after, pairs: list[tuple[str, str | None]]):
        """Insert (label, quote) paragraph pairs after paragraph `after` (a label alone when the
        quote is None, e.g. "Modified: see Comment 2."); returns the last paragraph inserted.
        Original labels and quotes are green and Modified ones blue, as in the letter."""
        for label, quote in pairs:
            color = ORIGINAL_GREEN if label.startswith("Original") else MODIFIED_BLUE
            lp = self._clone_paragraph(self._t_label, label)
            self._set_color(lp, color)
            after.addnext(lp)
            after = lp
            if quote is not None:
                qp = self._clone_paragraph(self._t_quote, quote)
                self._set_color(qp, color)
                lp.addnext(qp)
                after = qp
        return after
