"""Shared vocabulary. Domains double as the filter list for the 'find people' search."""
from __future__ import annotations

import re

DOMAINS = [
    "Backend",
    "Frontend",
    "Mobile (iOS)",
    "Mobile (Android)",
    "Cross-platform Mobile",
    "UI/UX Design",
    "Machine Learning",
    "Data Science & Analytics",
    "DevOps & Cloud",
    "Cybersecurity",
    "Embedded & IoT",
    "Game Development",
    "Product & Management",
    "Content & Marketing",
]

# first matching keyword wins, so list the more specific ones first
_DOMAIN_ALIASES = [
    ("ios", "Mobile (iOS)"), ("swift", "Mobile (iOS)"),
    ("android", "Mobile (Android)"), ("kotlin", "Mobile (Android)"),
    ("flutter", "Cross-platform Mobile"), ("react native", "Cross-platform Mobile"),
    ("cross-platform", "Cross-platform Mobile"), ("cross platform", "Cross-platform Mobile"),
    ("ui/ux", "UI/UX Design"), ("ux", "UI/UX Design"), ("ui", "UI/UX Design"),
    ("design", "UI/UX Design"), ("figma", "UI/UX Design"),
    ("machine learning", "Machine Learning"), ("deep learning", "Machine Learning"),
    ("ml", "Machine Learning"), ("ai", "Machine Learning"), ("nlp", "Machine Learning"),
    ("computer vision", "Machine Learning"), ("llm", "Machine Learning"),
    ("data", "Data Science & Analytics"), ("analytics", "Data Science & Analytics"),
    ("devops", "DevOps & Cloud"), ("cloud", "DevOps & Cloud"),
    ("security", "Cybersecurity"), ("cyber", "Cybersecurity"),
    ("embedded", "Embedded & IoT"), ("iot", "Embedded & IoT"), ("hardware", "Embedded & IoT"),
    ("game", "Game Development"),
    ("product", "Product & Management"), ("management", "Product & Management"),
    ("content", "Content & Marketing"), ("marketing", "Content & Marketing"),
    ("video", "Content & Marketing"),
    ("frontend", "Frontend"), ("front-end", "Frontend"), ("front end", "Frontend"), ("web", "Frontend"),
    ("backend", "Backend"), ("back-end", "Backend"), ("back end", "Backend"),
    ("server", "Backend"), ("api", "Backend"),
]


def normalize_domain(raw: str) -> str | None:
    """Maps free text like 'Deep Learning' or 'iOS dev' onto one of DOMAINS, or None."""
    text = (raw or "").strip().lower()
    if not text:
        return None
    for domain in DOMAINS:
        if text == domain.lower():
            return domain
    for keyword, domain in _DOMAIN_ALIASES:
        if re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text):
            return domain
    return None


# canonical skill name -> spellings people actually write on resumes
SKILL_ALIASES: dict[str, list[str]] = {
    "Python": ["python3", "py"],
    "JavaScript": ["js", "javascript (es6)", "es6", "ecmascript"],
    "TypeScript": ["ts"],
    "React": ["reactjs", "react.js"],
    "Node.js": ["nodejs", "node js", "node"],
    "Express.js": ["express", "expressjs"],
    "Next.js": ["nextjs", "next js"],
    "Vue.js": ["vue", "vuejs"],
    "Flask": [],
    "Django": [],
    "FastAPI": ["fast api"],
    "Java": [],
    "C++": ["cpp", "c plus plus"],
    "C": [],
    "C#": ["csharp", "c sharp"],
    "Go": ["golang"],
    "Swift": [],
    "SwiftUI": ["swift ui"],
    "Kotlin": [],
    "Dart": [],
    "Flutter": [],
    "HTML/CSS": ["html", "css", "html5", "css3", "html & css"],
    "Tailwind CSS": ["tailwind", "tailwindcss"],
    "SQL": [],
    "PostgreSQL": ["postgres", "postgresql"],
    "MySQL": [],
    "MongoDB": ["mongo", "mongo db"],
    "Firebase": [],
    "SQLite": [],
    "Redis": [],
    "REST APIs": ["rest", "restful", "rest api", "restful apis"],
    "GraphQL": [],
    "Docker": [],
    "Kubernetes": ["k8s"],
    "AWS": ["amazon web services"],
    "Google Cloud": ["gcp", "google cloud platform"],
    "Git/GitHub": ["git", "github"],
    "Linux": [],
    "TensorFlow": ["tf"],
    "PyTorch": ["torch"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "Pandas": [],
    "NumPy": ["numpy"],
    "OpenCV": ["cv2"],
    "NLP": ["natural language processing"],
    "LangChain": [],
    "LangGraph": [],
    "RAG": ["retrieval augmented generation", "retrieval-augmented generation"],
    "Machine Learning": ["ml"],
    "Deep Learning": ["dl"],
    "Figma": [],
    "Adobe XD": ["xd"],
    "UI/UX Design": ["ui design", "ux design", "ui/ux", "uiux"],
    "Unity": [],
    "Arduino": [],
    "Raspberry Pi": ["rpi"],
}

_ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _aliases in SKILL_ALIASES.items():
    _ALIAS_TO_CANON[_canon.lower()] = _canon
    for _alias in _aliases:
        _ALIAS_TO_CANON.setdefault(_alias.lower(), _canon)


def canonical_skill(raw: str) -> str:
    """'reactjs' -> 'React'. Unknown skills keep the user's spelling, trimmed."""
    cleaned = re.sub(r"\s+", " ", (raw or "").strip())
    return _ALIAS_TO_CANON.get(cleaned.lower(), cleaned)


def spellings(canonical: str) -> list[str]:
    """Every way the skill might be written in the resume text."""
    return [canonical] + SKILL_ALIASES.get(canonical, [])


# which domain a skill belongs to (used to measure how deep someone is in a domain)
SKILL_DOMAINS: dict[str, str] = {}
for _domain, _names in {
    "Backend": ["Flask", "Django", "FastAPI", "Node.js", "Express.js", "REST APIs", "GraphQL", "SQL",
                "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Java", "Go", "C#", "Firebase"],
    "Frontend": ["React", "Next.js", "Vue.js", "JavaScript", "TypeScript", "HTML/CSS", "Tailwind CSS"],
    "Mobile (iOS)": ["Swift", "SwiftUI", "Xcode"],
    "Mobile (Android)": ["Kotlin", "Jetpack Compose"],
    "Cross-platform Mobile": ["Flutter", "Dart", "React Native"],
    "UI/UX Design": ["Figma", "Adobe XD", "UI/UX Design", "Prototyping", "User Research", "Wireframing"],
    "Machine Learning": ["TensorFlow", "PyTorch", "scikit-learn", "NLP", "LangChain", "LangGraph", "RAG",
                         "Machine Learning", "Deep Learning", "OpenCV", "Computer Vision", "Keras",
                         "HuggingFace Transformers", "Prompt Engineering", "Vector Databases"],
    "Data Science & Analytics": ["Pandas", "NumPy", "Matplotlib", "Seaborn", "Power BI", "Tableau", "Data Analysis"],
    "DevOps & Cloud": ["Docker", "Kubernetes", "AWS", "Google Cloud", "Linux", "CI/CD", "Terraform"],
    "Embedded & IoT": ["Arduino", "Raspberry Pi", "Embedded C"],
    "Game Development": ["Unity", "Unreal Engine"],
    "Content & Marketing": ["Video Editing", "Copywriting", "Social Media", "SEO"],
}.items():
    for _n in _names:
        SKILL_DOMAINS[_n.lower()] = _domain


def skill_domain(name: str) -> str | None:
    """'SwiftUI' -> 'Mobile (iOS)'; 'TensorFlow/Keras' -> 'Machine Learning'; unknown -> None."""
    canon = canonical_skill(name).lower()
    if canon in SKILL_DOMAINS:
        return SKILL_DOMAINS[canon]
    tokens = set(re.findall(r"[a-z0-9+#.]+", canon))
    for key, domain in SKILL_DOMAINS.items():
        if key in tokens or key in canon.split("/"):
            return domain
    return normalize_domain(name)
