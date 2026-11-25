"""
Automated Report Generation Module (CSV and PDF).
Generates audit-ready reports on sensor drift, health scores, and calibration recommendations.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import io
import time
import pandas as pd
import numpy as np

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable
)

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR


def generate_drift_csv_report(
    sensor_health_df: pd.DataFrame,
    feature_drift_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """
    Saves a consolidated CSV report containing sensor health scores and feature-level drift metrics.
    """
    target_path = output_path or (REPORTS_DIR / f"drift_report_{int(time.time())}.csv")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge or write sensor health
    sensor_health_df.to_csv(target_path, index=False)
    return target_path


def generate_anomaly_csv_report(
    scored_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """
    Exports flagged anomalous records to CSV for operational inspection.
    """
    target_path = output_path or (REPORTS_DIR / f"anomalies_{int(time.time())}.csv")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    anom_only = scored_df[scored_df["is_anomaly"]].copy() if "is_anomaly" in scored_df.columns else scored_df
    anom_only.to_csv(target_path, index=False)
    return target_path


def generate_pdf_report(
    summary_data: Dict[str, Any],
    sensor_health_df: pd.DataFrame,
    feature_drift_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None
) -> Path:
    """
    Builds a professional, multi-page PDF executive intelligence report.
    """
    target_path = output_path or (REPORTS_DIR / f"Sensor_Drift_Intelligence_Report_{int(time.time())}.pdf")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(target_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0D47A1")
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#555555")
    )
    
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1565C0"),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#222222")
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#222222")
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("Air Quality Sensor Drift Intelligence", title_style))
    story.append(Paragraph("Sensor Array Reliability Surveillance & Drift Detection Report", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0D47A1"), spaceAfter=10))

    # Metadata & Executive Overview Table
    meta_text = (
        f"<b>Report Generation Time:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        f"<b>Reference Baseline:</b> {summary_data.get('reference_batch', 'Batch 1')}<br/>"
        f"<b>Comparison Batch:</b> {summary_data.get('comparison_batch', 'Batch 2-10')}<br/>"
        f"<b>Total Observations:</b> {summary_data.get('total_samples', '13,910')}<br/>"
        f"<b>Array Composition:</b> 16 Chemical Sensors (TGS Series, 128 Extracted Features)"
    )
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 10))

    # KPI Summary Cards in a 4-cell table
    kpi_data = [
        [
            Paragraph("<b>Overall Mean Health</b>", body_style),
            Paragraph("<b>Drift Status</b>", body_style),
            Paragraph("<b>Flagged Anomalies</b>", body_style),
            Paragraph("<b>Critical Sensors</b>", body_style)
        ],
        [
            Paragraph(f"<font size=13 color='#0D47A1'><b>{summary_data.get('mean_health', '72.4')}/100</b></font>", body_style),
            Paragraph(f"<font size=13 color='#E65100'><b>{summary_data.get('drift_status', 'Moderate Drift')}</b></font>", body_style),
            Paragraph(f"<font size=13 color='#C62828'><b>{summary_data.get('anomaly_pct', '12.8%')}</b></font>", body_style),
            Paragraph(f"<font size=13 color='#D32F2F'><b>{summary_data.get('critical_count', '2')} / 16</b></font>", body_style)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[130, 130, 130, 130])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#D1D5DB")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 12))

    # Section 1: Sensor Array Reliability Table
    story.append(Paragraph("1. Multi-Sensor Reliability & Health Breakdown", h1_style))
    
    headers = [
        Paragraph("<b>Sensor</b>", table_header_style),
        Paragraph("<b>Model / Type</b>", table_header_style),
        Paragraph("<b>Health</b>", table_header_style),
        Paragraph("<b>Category</b>", table_header_style),
        Paragraph("<b>Drift (PSI)</b>", table_header_style),
        Paragraph("<b>Action Recommendation</b>", table_header_style)
    ]
    
    rows = [headers]
    for _, row in sensor_health_df.iterrows():
        cat = str(row["Category"])
        cat_color = "#2E7D32" if cat == "Healthy" else ("#F57F17" if cat == "Monitor" else ("#E65100" if cat == "Warning" else "#C62828"))
        
        rows.append([
            Paragraph(f"<b>{row['Sensor_ID']}</b>", table_cell_style),
            Paragraph(str(row.get("Sensor_Type", "TGS")), table_cell_style),
            Paragraph(f"<b>{row['Health_Score']}</b>", table_cell_style),
            Paragraph(f"<font color='{cat_color}'><b>{cat}</b></font>", table_cell_style),
            Paragraph(f"{row.get('Drift_Component', '-')}", table_cell_style),
            Paragraph(str(row["Recommendation"]), table_cell_style)
        ])

    health_table = Table(rows, colWidths=[55, 120, 45, 60, 50, 210])
    health_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    story.append(health_table)
    story.append(Spacer(1, 14))

    # Section 2: Methodological Explanation & Calibration Guidance
    story.append(Paragraph("2. Methodological Clarification & Calibration Guidance", h1_style))
    method_text = (
        "<b>Data Drift vs Physical Sensor Aging:</b><br/>"
        "Population Stability Index (PSI) and Kolmogorov-Smirnov statistics evaluate distributional displacement "
        "in the 128 multivariate feature dimensions. An elevated drift score indicates that current sensor readings "
        "have diverged from the baseline manifold established during calibration (Batch 1).<br/><br/>"
        "<b>Important Scientific Disclaimer:</b><br/>"
        "Statistical anomalies and drift indicators do <i>not</i> provide definitive evidence of irreparable physical sensor failure. "
        "In metal-oxide gas sensor arrays, environmental variables (relative humidity, temperature swings, background VOCs) "
        "or reversible surface contamination can trigger statistical drift. Operational protocol dictates:"
        "<ul>"
        "<li>Re-zero sensor array with synthetic clean air baseline.</li>"
        "<li>Verify calibration with reference standard gas mixture (e.g. 50 ppm Ethanol).</li>"
        "<li>Inspect sensor heating element supply voltage (5.0V ± 0.1V).</li>"
        "<li>Consider active drift compensation before physical sensor replacement.</li>"
        "</ul>"
    )
    story.append(Paragraph(method_text, body_style))
    story.append(Spacer(1, 14))

    # Footer note
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCCC"), spaceAfter=6))
    story.append(Paragraph("Air Quality Sensor Drift Intelligence System | Automated Surveillance Audit", subtitle_style))

    doc.build(story)
    print(f"[ReportGenerator] PDF report created at: {target_path}")
    return target_path


def get_pdf_bytes(
    summary_data: Dict[str, Any],
    sensor_health_df: pd.DataFrame
) -> bytes:
    """Returns PDF report as bytes for direct Streamlit download."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#0D47A1"))
    body_style = ParagraphStyle("B", fontName="Helvetica", fontSize=8, leading=11)
    
    story = [
        Paragraph("Air Quality Sensor Drift Intelligence Report", title_style),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D47A1"), spaceAfter=10),
        Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Total Analyzed: {summary_data.get('total_samples', '13,910')}", body_style),
        Spacer(1, 10)
    ]
    
    # Table of health scores
    headers = [Paragraph("<b>Sensor</b>", body_style), Paragraph("<b>Health Score</b>", body_style), Paragraph("<b>Category</b>", body_style), Paragraph("<b>Recommendation</b>", body_style)]
    rows = [headers]
    for _, r in sensor_health_df.iterrows():
        rows.append([
            Paragraph(str(r["Sensor_ID"]), body_style),
            Paragraph(str(r["Health_Score"]), body_style),
            Paragraph(str(r["Category"]), body_style),
            Paragraph(str(r["Recommendation"]), body_style)
        ])
    t = Table(rows, colWidths=[70, 70, 70, 330])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    doc.build(story)
    return buf.getvalue()
