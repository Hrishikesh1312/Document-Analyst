from __future__ import annotations

import html
import io
import re
from typing import Any


def conversation_markdown(
    name: str,
    messages: list[dict[str, str]],
    sources_by_turn: dict[int, list[Any]],
) -> str:
    lines = [f"# {name}", ""]
    for index, message in enumerate(messages):
        role = "User" if message.get("role") == "user" else "Assistant"
        lines.extend([f"## {role}", "", message.get("content", "").strip(), ""])
        sources = sources_by_turn.get(index, [])
        if sources:
            lines.append("Sources:")
            for source in sources:
                lines.append(
                    f"- [{source.source_id}] {source.document_name}, page "
                    f"{source.approx_page} - {source.source_path}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def conversation_pdf(
    name: str,
    messages: list[dict[str, str]],
    sources_by_turn: dict[int, list[Any]],
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    output = io.BytesIO()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ConversationTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=27,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#123047"),
        spaceAfter=12,
    )
    role_style = ParagraphStyle(
        "Role",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#147A63"),
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#172B3A"),
        spaceAfter=6,
    )
    source_style = ParagraphStyle(
        "Source",
        parent=body_style,
        fontSize=8,
        leading=11,
        leftIndent=8,
        textColor=colors.HexColor("#4B6475"),
    )

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6B7F8D"))
        canvas.drawString(20 * mm, 12 * mm, "Document Analyst conversation export")
        canvas.drawRightString(190 * mm, 12 * mm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=name,
        author="Document Analyst",
    )
    story = [Paragraph(html.escape(name), title_style), Spacer(1, 4 * mm)]
    for index, message in enumerate(messages):
        role = "User" if message.get("role") == "user" else "Assistant"
        story.append(Paragraph(role, role_style))
        content = _plain_text(message.get("content", ""))
        paragraphs = content.split("\n\n") or [""]
        for paragraph in paragraphs:
            story.append(Paragraph(html.escape(paragraph).replace("\n", "<br/>"), body_style))
        sources = sources_by_turn.get(index, [])
        if sources:
            story.append(Paragraph("Sources", role_style))
            for source in sources:
                label = (
                    f"[{source.source_id}] {source.document_name}, page "
                    f"{source.approx_page} - {source.source_path}"
                )
                story.append(Paragraph(html.escape(label), source_style))
        story.append(Spacer(1, 2 * mm))
    if not messages:
        story.append(Paragraph("This conversation has no messages yet.", body_style))
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


def _plain_text(value: str) -> str:
    text = re.sub(r"```(?:\w+)?\n(.*?)```", r"\1", value, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text.strip()
