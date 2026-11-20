"""PDF forensic reports and CSV exports.

Reports contain only values stored from the existing model's response.  Fields
the model does not return (for example a separate genuine score) are shown as
"Not returned by model" instead of being invented.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#09090B")
MUTED = colors.HexColor("#71717A")
LINE = colors.HexColor("#E4E4E7")
SOFT = colors.HexColor("#FAFAFA")
BONAFIDE = colors.HexColor("#22C55E")
SPOOF = colors.HexColor("#F43F5E")
UNCERTAIN = colors.HexColor("#F59E0B")

VERDICT_COLORS = {"bonafide": BONAFIDE, "spoof": SPOOF, "uncertain": UNCERTAIN}

CSV_COLUMNS = [
    ("report_id", "Report ID"),
    ("created_at", "Analysed at (UTC)"),
    ("filename", "Filename"),
    ("file_format", "Format"),
    ("size_bytes", "Size (bytes)"),
    ("duration_seconds", "Duration (s)"),
    ("sample_rate", "Sample rate (Hz)"),
    ("channels", "Channels"),
    ("prediction", "Prediction"),
    ("spoof_score", "Spoof score (0-1)"),
    ("raw_spoof_score", "Raw spoof score (0-1)"),
    ("confidence", "Confidence (0-1)"),
    ("confidence_label", "Confidence label"),
    ("chunk_count", "Chunks analysed"),
    ("chunk_seconds", "Chunk length (s)"),
    ("processing_ms", "Processing time (ms)"),
    ("model_name", "Model"),
    ("feature_set", "Feature set"),
    ("n_features", "Feature count"),
]


def _fmt_dt(value: datetime | None) -> str:
    if not value:
        return "Not available"
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.strftime("%d %b %Y, %H:%M:%S UTC")


def _fmt(value, suffix: str = "") -> str:
    if value is None or value == "":
        return "Not available"
    if isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    return f"{text}{suffix}"


def _pct(value) -> str:
    if value is None:
        return "Not available"
    return f"{float(value) * 100:.2f}%"


def _draw_mark(c: pdfcanvas.Canvas, x: float, y: float, size: float, color=INK) -> None:
    """Vector version of the VeriHuman shield + waveform mark."""
    c.saveState()
    c.translate(x, y)
    scale = size / 24.0
    c.scale(scale, scale)
    c.setStrokeColor(color)
    c.setLineWidth(1.7)
    c.setLineCap(1)
    c.setLineJoin(1)

    p = c.beginPath()
    p.moveTo(12, 1.6)
    p.lineTo(22.4, 5.4)
    p.lineTo(22.4, 13.0)
    p.curveTo(22.4, 18.6, 18.3, 22.6, 12, 24.0)
    p.curveTo(5.7, 22.6, 1.6, 18.6, 1.6, 13.0)
    p.lineTo(1.6, 5.4)
    p.close()
    c.drawPath(p, stroke=1, fill=0)

    bars = [(6.4, 11.0, 1.4), (9.0, 8.6, 1.4), (11.6, 6.2, 1.4),
            (14.2, 8.6, 1.4), (16.8, 11.0, 1.4)]
    for bx, by, bh in bars:
        c.line(bx, by, bx, by + bh)
    c.restoreState()


def _verdict_text(analysis) -> str:
    return {
        "bonafide": "Human Voice Detected",
        "spoof": "Potential AI-Generated or Manipulated Voice",
        "uncertain": "Inconclusive Audio Analysis",
    }.get(analysis.verdict_slug, "Analysis Result")


def build_pdf(analysis, user, info: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=30 * mm,
        bottomMargin=20 * mm,
        title=f"VeriHuman report {analysis.report_id}",
        author="VeriHuman",
        subject="Audio authenticity analysis report",
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "VHBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.2, leading=14, textColor=INK, alignment=TA_LEFT,
    )
    small = ParagraphStyle(
        "VHSmall", parent=body, fontSize=8, leading=11.5, textColor=MUTED,
    )
    h2 = ParagraphStyle(
        "VHH2", parent=body, fontName="Helvetica-Bold", fontSize=11.5,
        leading=14, spaceBefore=10, spaceAfter=6,
    )
    verdict_style = ParagraphStyle(
        "VHVerdict", parent=body, fontName="Helvetica-Bold", fontSize=15,
        leading=19, textColor=colors.white,
    )
    verdict_sub = ParagraphStyle(
        "VHVerdictSub", parent=body, fontSize=8.8, leading=12,
        textColor=colors.Color(1, 1, 1, 0.86),
    )

    verdict_color = VERDICT_COLORS.get(analysis.verdict_slug, MUTED)
    verdict = Table(
        [[
            Paragraph(
                f"{analysis.prediction}<br/><font size=10>{_verdict_text(analysis)}</font>",
                verdict_style,
            ),
            Paragraph(
                f"Spoof score<br/><b>{analysis.spoof_percent}</b><br/>"
                f"Chunks: {_fmt(analysis.chunk_count)}",
                verdict_sub,
            ),
        ]],
        colWidths=[118 * mm, 56 * mm],
    )
    verdict.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), verdict_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 12),
        ("RIGHTPADDING", (0, 0), (0, 0), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))

    def kv_table(rows, widths=(55 * mm, 119 * mm)):
        table = Table(
            [[Paragraph(f"<b>{key}</b>", small), Paragraph(str(value), body)] for key, value in rows],
            colWidths=list(widths),
        )
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return table

    file_rows = [
        ("Filename", analysis.filename),
        ("Format", analysis.file_format),
        ("File size", analysis.size_label),
        ("Duration", analysis.duration_label),
        ("Sample rate", _fmt(analysis.sample_rate, " Hz")),
        ("Channels", _fmt(analysis.channels)),
        ("Analysed at", _fmt_dt(analysis.created_at)),
        ("Processing time", _fmt(analysis.processing_ms, " ms")),
        ("Chunks analysed", _fmt(analysis.chunk_count)),
        ("Chunk length", _fmt(analysis.chunk_seconds, " s")),
    ]
    score_rows = [
        ("Spoof score (calibrated)", analysis.spoof_percent),
        ("Raw model score", _pct(analysis.raw_spoof_score)),
        ("Genuine score", "Not returned by model"),
        ("Confidence", f"{analysis.confidence_percent} ({analysis.confidence_label or 'not provided'})"),
        ("Spoof threshold", _pct(analysis.spoof_threshold)),
        ("Bonafide threshold", _pct(analysis.bonafide_threshold)),
        ("Chunk disagreement", _pct(analysis.chunk_disagreement)),
        ("Decision reason", analysis.reason or "Not provided"),
    ]
    model_rows = [
        ("Model", analysis.model_name or info.get("model_name", "Not available")),
        ("Model created", analysis.model_version or "Not available"),
        ("Feature set", analysis.feature_set or info.get("feature_set", "Not available")),
        ("Feature count", _fmt(analysis.n_features)),
        ("Calibration", "Platt scaling" if info.get("calibrated") else "Not calibrated"),
        ("Training corpus", "ASVspoof 2019 Logical Access"),
    ]

    story = [
        Spacer(1, 4),
        Paragraph("File and analysis metadata", h2),
        kv_table(file_rows),
        Spacer(1, 8),
        Paragraph("Detection scores", h2),
        kv_table(score_rows),
        Spacer(1, 8),
        Paragraph("Model information", h2),
        kv_table(model_rows),
    ]

    chunks = analysis.chunks
    if chunks:
        header = ["#", "Start (s)", "Length (s)", "Spoof score", "Weight"]
        data = [header]
        for index, chunk in enumerate(chunks[:14], start=1):
            data.append([
                str(index),
                _fmt(chunk.get("start_seconds")),
                _fmt(chunk.get("duration_seconds")),
                _fmt(chunk.get("spoof_score"), "%"),
                _fmt(chunk.get("weight")),
            ])
        table = Table(data, colWidths=[12 * mm, 26 * mm, 26 * mm, 34 * mm, 30 * mm])
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 1), (-1, -1), INK),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(Paragraph("Chunk-level results", h2))
        story.append(table)
        if len(chunks) > 14:
            story.append(Paragraph(
                f"{len(chunks) - 14} further chunk(s) are stored in the website history.",
                small,
            ))

    story.extend([
        Spacer(1, 10),
        Paragraph("Technical explanation", h2),
        Paragraph(
            "The recording was decoded to mono audio, split into 5-second windows, and each "
            "window was converted into the project's existing 172-feature vector (MFCCs, delta "
            "and delta-delta coefficients, spectral shape, flatness, flux, contrast, chroma, "
            "zero-crossing rate and RMS). The existing trained classifier returned one spoof "
            "score per window. Window scores were combined with duration and energy weighting, "
            "the result was calibrated with Platt scaling fitted on held-out development "
            "speakers, and the final decision used the project's spoof and bonafide thresholds.",
            body,
        ),
        Spacer(1, 6),
        Paragraph("Detection limitations", h2),
        Paragraph(
            "This report is an automated estimate, not definitive proof of authenticity. The "
            "model is trained on the ASVspoof 2019 Logical Access corpus and has not been "
            "validated on arbitrary internet audio, music, singing, heavy noise, telephone-band "
            "speech or deliberately adversarial manipulations. Some attack families remain hard "
            "to detect. An UNCERTAIN result means the score fell between the two thresholds and "
            "needs human review. Codec and resampling changes can move scores, so compare files "
            "recorded and encoded in similar conditions.",
            body,
        ),
        Spacer(1, 8),
        Paragraph(
            f"Report {analysis.report_id} · Generated for {user.name} ({user.email}) · "
            f"VeriHuman {datetime.now(timezone.utc).year}",
            small,
        ),
    ])

    header_info = {
        "report_id": analysis.report_id,
        "created": _fmt_dt(analysis.created_at),
    }

    def decorate(c: pdfcanvas.Canvas, _doc) -> None:
        c.saveState()
        page_width, page_height = A4
        _draw_mark(c, 18 * mm, page_height - 22 * mm, 9 * mm, INK)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(30 * mm, page_height - 19.2 * mm, "VeriHuman")
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.5)
        c.drawString(30 * mm, page_height - 23.2 * mm, "Trust Every Voice. Verify Every Sound.")
        c.setFont("Helvetica-Bold", 8.5)
        c.setFillColor(INK)
        c.drawRightString(page_width - 18 * mm, page_height - 17 * mm, "FORENSIC AUDIO REPORT")
        c.setFont("Helvetica", 8)
        c.setFillColor(MUTED)
        c.drawRightString(page_width - 18 * mm, page_height - 21 * mm, header_info["report_id"])
        c.drawRightString(page_width - 18 * mm, page_height - 25 * mm, header_info["created"])
        c.setStrokeColor(LINE)
        c.setLineWidth(0.6)
        c.line(18 * mm, page_height - 28 * mm, page_width - 18 * mm, page_height - 28 * mm)

        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.2)
        c.drawString(
            18 * mm, 12 * mm,
            "Automated estimate — not definitive proof of authenticity.",
        )
        c.drawRightString(page_width - 18 * mm, 12 * mm, f"Page {c.getPageNumber()}")
        c.restoreState()

    doc.build([verdict, Spacer(1, 12), *story], onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()


def build_csv(analyses) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([label for _key, label in CSV_COLUMNS])
    for analysis in analyses:
        row = []
        for key, _label in CSV_COLUMNS:
            if key == "report_id":
                row.append(analysis.report_id)
            elif key == "created_at":
                row.append(_fmt_dt(analysis.created_at))
            else:
                row.append(getattr(analysis, key, ""))
        writer.writerow(row)
    # UTF-8 BOM keeps Excel happy with non-ASCII filenames.
    return "\ufeff" + output.getvalue()
