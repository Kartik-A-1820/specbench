/*
 * scripts/render_kt_docx.js — render the KT document model into a .docx.
 *
 * Reads the JSON document model produced by scripts/build_kt_docx.py and writes
 * a Word document. Kept deliberately dumb: all content decisions, and every
 * number, are made on the Python side against the measured results. This file
 * only knows how to draw.
 *
 * Usage:
 *   node scripts/render_kt_docx.js build/kt_model.json out.docx
 */

const fs = require("fs");
const {
  AlignmentType,
  BorderStyle,
  Document,
  HeadingLevel,
  LevelFormat,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} = require("docx");

const [, , modelPath, outPath] = process.argv;
if (!modelPath || !outPath) {
  console.error("usage: node render_kt_docx.js <model.json> <out.docx>");
  process.exit(1);
}

const model = JSON.parse(fs.readFileSync(modelPath, "utf8"));

// US Letter, one inch margins. Content width in DXA: 12240 - 2*1440 = 9360.
const PAGE_WIDTH = 12240;
const PAGE_HEIGHT = 15840;
const MARGIN = 1440;
const CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN;

const MONO = "Consolas";
const BODY = "Calibri";

/* Split a string on inline markdown: **bold** and `code`. */
function inlineRuns(text, baseOpts = {}) {
  const runs = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      runs.push(new TextRun({ text: text.slice(last, m.index), ...baseOpts }));
    }
    const tok = m[0];
    if (tok.startsWith("**")) {
      runs.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...baseOpts }));
    } else {
      runs.push(
        new TextRun({ text: tok.slice(1, -1), font: MONO, size: 19, ...baseOpts })
      );
    }
    last = m.index + tok.length;
  }
  if (last < text.length) {
    runs.push(new TextRun({ text: text.slice(last), ...baseOpts }));
  }
  return runs.length ? runs : [new TextRun({ text: "", ...baseOpts })];
}

function makeTable(rows) {
  const nCols = Math.max(...rows.map((r) => r.length));
  const colWidth = Math.floor(CONTENT_WIDTH / nCols);
  // Column widths must sum to the table width, so absorb the remainder.
  const widths = new Array(nCols).fill(colWidth);
  widths[nCols - 1] = CONTENT_WIDTH - colWidth * (nCols - 1);

  const tableRows = rows.map((cells, rowIdx) => {
    const isHeader = rowIdx === 0;
    const padded = [...cells];
    while (padded.length < nCols) padded.push("");
    return new TableRow({
      tableHeader: isHeader,
      children: padded.map(
        (cell, colIdx) =>
          new TableCell({
            width: { size: widths[colIdx], type: WidthType.DXA },
            shading: isHeader
              ? { type: ShadingType.CLEAR, fill: "E7EDF4", color: "auto" }
              : undefined,
            margins: { top: 60, bottom: 60, left: 110, right: 110 },
            children: [
              new Paragraph({
                spacing: { before: 0, after: 0 },
                children: inlineRuns(cell, {
                  bold: isHeader,
                  size: 19,
                  font: BODY,
                }),
              }),
            ],
          })
      ),
    });
  });

  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    rows: tableRows,
  });
}

const children = [];

/* Title block. */
children.push(
  new Paragraph({
    spacing: { before: 2400, after: 120 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: model.title, bold: true, size: 44, font: BODY })],
  })
);
children.push(
  new Paragraph({
    spacing: { after: 360 },
    alignment: AlignmentType.CENTER,
    children: [
      new TextRun({ text: model.subtitle, size: 26, color: "555555", font: BODY }),
    ],
  })
);
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [
      new TextRun({
        text: `Generated ${model.generated}`,
        size: 19,
        color: "888888",
        font: BODY,
      }),
    ],
  })
);
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [
      new TextRun({
        text:
          "Every figure in the results sections is generated directly from " +
          "results/raw_measurements.jsonl and results/summary.json. Anything not " +
          "measured is marked NOT MEASURED rather than estimated.",
        size: 19,
        italics: true,
        color: "888888",
        font: BODY,
      }),
    ],
  })
);
// No explicit break here: every "PART ..." heading already carries
// pageBreakBefore, so adding one would leave a blank page after the title.

const HEADING_MAP = {
  heading1: HeadingLevel.HEADING_1,
  heading2: HeadingLevel.HEADING_1,
  heading3: HeadingLevel.HEADING_2,
  heading4: HeadingLevel.HEADING_3,
};

for (const block of model.blocks) {
  switch (block.type) {
    case "heading1":
    case "heading2":
    case "heading3":
    case "heading4": {
      const isTop = block.type === "heading1" || block.type === "heading2";
      children.push(
        new Paragraph({
          heading: HEADING_MAP[block.type],
          pageBreakBefore: isTop && /^PART /.test(block.text),
          spacing: { before: isTop ? 360 : 260, after: 140 },
          children: [
            new TextRun({
              text: block.text,
              bold: true,
              font: BODY,
              size: isTop ? 30 : block.type === "heading3" ? 25 : 22,
              color: isTop ? "1F3864" : "2E5395",
            }),
          ],
        })
      );
      break;
    }

    case "para":
      children.push(
        new Paragraph({
          spacing: { before: 60, after: 140, line: 288 },
          children: inlineRuns(block.text, { size: 21, font: BODY }),
        })
      );
      break;

    case "list":
      for (const item of block.items) {
        children.push(
          new Paragraph({
            numbering: { reference: "kt-bullets", level: 0 },
            spacing: { before: 40, after: 40, line: 276 },
            children: inlineRuns(item, { size: 21, font: BODY }),
          })
        );
      }
      break;

    case "code":
      for (const codeLine of block.lines) {
        children.push(
          new Paragraph({
            spacing: { before: 0, after: 0 },
            indent: { left: 360 },
            shading: { type: ShadingType.CLEAR, fill: "F4F6F8", color: "auto" },
            children: [
              new TextRun({ text: codeLine || " ", font: MONO, size: 19 }),
            ],
          })
        );
      }
      children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      break;

    case "table":
      children.push(makeTable(block.rows));
      children.push(new Paragraph({ spacing: { after: 180 }, children: [] }));
      break;

    case "rule":
      children.push(
        new Paragraph({
          spacing: { before: 120, after: 120 },
          border: {
            bottom: { style: BorderStyle.SINGLE, size: 6, color: "CCCCCC", space: 1 },
          },
          children: [],
        })
      );
      break;

    default:
      break;
  }
}

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "kt-bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 460, hanging: 260 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
          margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
        },
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outPath, buf);
  console.log(`Wrote ${outPath} (${(buf.length / 1024).toFixed(1)} KB, ${children.length} elements)`);
});
