"""Creates three FICTIONAL sample resume PDFs in backend/samples/ for testing.

    python scripts/make_sample_resumes.py
"""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

RESUMES = {
    "rohan_mehta_backend": """Rohan Mehta
B.Tech Computer Science, 3rd Year | Sample Institute of Technology

SKILLS
Python, Flask, Django, PostgreSQL, MongoDB, REST APIs, Docker, Git, Redis, AWS

PROJECTS
Campus Cab Share - Flask, PostgreSQL, Docker. Built REST APIs for ride matching used by 300 students.
Student Marketplace - Django, Redis. Designed a caching layer that cut page load time by 40 percent.
Expense Tracker API - Flask, MongoDB. Added JWT authentication and unit tests.

EXPERIENCE
Backend Intern, FinEdge Labs (Jun-Aug 2025): built microservices in Python and FastAPI, deployed on AWS EC2 with Docker.

AWARDS
Winner, internal hackathon 2025 (backend track).
""",
    "ananya_rao_design": """Ananya Rao
B.Tech Computer Science, 2nd Year | Sample Institute of Technology

SKILLS
Figma, Adobe XD, UI/UX Design, Prototyping, User Research, HTML, CSS, Tailwind CSS, Swift (basics)

PROJECTS
Mess Menu App Redesign - Figma. Ran 12 user interviews and redesigned the ordering flow; usability score rose from 61 to 84.
Event Discovery App - Figma, Adobe XD. Clickable prototype with 18 screens, tested with 25 students.
Sustainability Dashboard - HTML, CSS, Tailwind CSS. Built the responsive front end for a campus energy dashboard.
iOS Habit Tracker - Swift. Completed a 4-week SwiftUI course project with local notifications.

EXPERIENCE
Design Intern, BrightCart Studio (Jan-Mar 2026): produced wireframes and a design system for a checkout flow.

ACHIEVEMENTS
Runner-up, college UI design contest 2025.
""",
    "kabir_sethi_ml": """Kabir Sethi
B.Tech Computer Science (AI), 3rd Year | Sample Institute of Technology

SKILLS
Python, PyTorch, scikit-learn, Pandas, NumPy, OpenCV, NLP, LangChain, RAG, SQL, Git

PROJECTS
Crop Disease Detector - PyTorch, OpenCV. Trained a CNN on 20k leaf images and reached 93 percent accuracy.
Resume Ranker - Python, scikit-learn, NLP. Ranked 500 resumes against job descriptions using TF-IDF and embeddings.
Campus FAQ Chatbot - LangChain, RAG, Python. Retrieval-augmented chatbot over college circulars.
Sales Forecasting - Pandas, NumPy. Time-series forecasting for a local retail dataset.

EXPERIENCE
ML Intern, DataNest (May-Jul 2025): cleaned datasets with Pandas and trained gradient-boosted models in scikit-learn.

PUBLICATIONS
Co-author, student conference paper on lightweight plant-disease classifiers.
""",
}


def make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()  # A4
    rect = fitz.Rect(48, 48, page.rect.width - 48, page.rect.height - 48)
    leftover = page.insert_textbox(rect, text, fontsize=10.5, fontname="helv")
    if leftover < 0:
        raise ValueError("sample text does not fit on one page")
    data = doc.tobytes()
    doc.close()
    return data


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "samples"
    out.mkdir(exist_ok=True)
    for name, text in RESUMES.items():
        (out / f"{name}.pdf").write_bytes(make_pdf(text))
        print("wrote", out / f"{name}.pdf")


if __name__ == "__main__":
    main()
