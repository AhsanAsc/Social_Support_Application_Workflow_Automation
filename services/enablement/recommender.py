from typing import List, Dict


def recommend_enablement_paths(applicant_profile: Dict) -> List[Dict]:
    """
    Given applicant profile (skills, work history, interests),
    return recommended upskilling and job opportunities.
    """
    # TODO: integrate with embeddings + a small rules/ML system
    return [
        {
            "type": "course",
            "title": "Basic Financial Literacy Workshop",
            "provider": "Gov Skills Center",
        },
        {
            "type": "job",
            "title": "Customer Support Associate (Entry-Level)",
            "employer": "Example Employer",
        },
    ]
