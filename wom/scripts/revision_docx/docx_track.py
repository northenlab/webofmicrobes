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
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
PROTECTED = {"hyperlink", "fldSimple", "sdt", "smartTag"}


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
        p = self.find(snippet, exact)
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

    def add_pairs(self, after, pairs: list[tuple[str, str]]):
        """Insert (label, quote) paragraph pairs after paragraph `after`; returns the last one."""
        for label, quote in pairs:
            lp = self._clone_paragraph(self._t_label, label)
            after.addnext(lp)
            qp = self._clone_paragraph(self._t_quote, quote)
            lp.addnext(qp)
            after = qp
        return after
