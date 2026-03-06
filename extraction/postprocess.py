"""Skill normalization and post-processing."""

SKILL_ALIASES: dict[str, str] = {
    "node": "Node.js", "nodejs": "Node.js", "node.js": "Node.js",
    "k8s": "Kubernetes", "kubernetes": "Kubernetes",
    "postgres": "PostgreSQL", "pg": "PostgreSQL", "postgresql": "PostgreSQL",
    "tf": "TensorFlow", "tensorflow": "TensorFlow", "tensorflow2": "TensorFlow",
    "py": "Python", "python3": "Python", "python": "Python",
    "js": "JavaScript", "javascript": "JavaScript", "es6": "JavaScript",
    "ts": "TypeScript", "typescript": "TypeScript",
    "react.js": "React", "reactjs": "React", "react": "React",
    "vue.js": "Vue.js", "vuejs": "Vue.js",
    "next.js": "Next.js", "nextjs": "Next.js",
    "c++": "C++", "cpp": "C++",
    "c#": "C#", "csharp": "C#",
    "go": "Go", "golang": "Go",
    "aws": "AWS", "amazon web services": "AWS",
    "gcp": "GCP", "google cloud": "GCP", "google cloud platform": "GCP",
    "azure": "Azure", "microsoft azure": "Azure",
    "docker": "Docker", "containerization": "Docker",
    "pytorch": "PyTorch", "torch": "PyTorch",
    "scikit-learn": "scikit-learn", "sklearn": "scikit-learn",
    "mongo": "MongoDB", "mongodb": "MongoDB",
    "redis": "Redis",
    "mysql": "MySQL",
    "graphql": "GraphQL",
    "rest": "REST", "rest api": "REST", "restful": "REST",
    "ci/cd": "CI/CD", "cicd": "CI/CD",
    "ml": "Machine Learning", "machine learning": "Machine Learning",
    "dl": "Deep Learning", "deep learning": "Deep Learning",
    "nlp": "NLP", "natural language processing": "NLP",
    "cv": "Computer Vision", "computer vision": "Computer Vision",
    "llm": "LLM", "large language model": "LLM", "large language models": "LLM",
    "java": "Java",
    "ruby": "Ruby",
    "rails": "Ruby on Rails", "ruby on rails": "Ruby on Rails",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "spring": "Spring", "spring boot": "Spring Boot",
    "kafka": "Kafka", "apache kafka": "Kafka",
    "spark": "Spark", "apache spark": "Spark", "pyspark": "Spark",
    "airflow": "Airflow", "apache airflow": "Airflow",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "jenkins": "Jenkins",
    "git": "Git", "github": "GitHub", "gitlab": "GitLab",
    "linux": "Linux",
    "sql": "SQL",
    "nosql": "NoSQL",
    "elasticsearch": "Elasticsearch", "elastic": "Elasticsearch",
    "rabbitmq": "RabbitMQ",
    "rust": "Rust",
    "scala": "Scala",
    "swift": "Swift",
    "kotlin": "Kotlin",
    "r": "R",
    "matlab": "MATLAB",
    "tableau": "Tableau",
    "power bi": "Power BI", "powerbi": "Power BI",
    "snowflake": "Snowflake",
    "databricks": "Databricks",
    "dbt": "dbt",
    "figma": "Figma",
    "sketch": "Sketch",
    "jira": "Jira",
    "confluence": "Confluence",
}


def normalize_skill(skill: str) -> str:
    """Normalize a single skill name."""
    key = skill.strip().lower()
    return SKILL_ALIASES.get(key, skill.strip())


def normalize_skills(skills: list[str]) -> list[str]:
    """Normalize and deduplicate a list of skills."""
    seen: set[str] = set()
    result: list[str] = []
    for s in skills:
        normalized = normalize_skill(s)
        if normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result


def build_skill_vocabulary(
    all_jobs_skills: list[list[str]], min_frequency: int = 3
) -> dict[str, int]:
    """Build vocabulary from all jobs, filtering by min frequency."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for skills in all_jobs_skills:
        for s in skills:
            counts[normalize_skill(s)] += 1
    return {
        skill: count
        for skill, count in counts.items()
        if count >= min_frequency
    }
