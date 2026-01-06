"""
Prompts centralisés pour le module d'enrichissement Vocalyx.

Objectifs :
- Avoir un point unique de définition des prompts (facile à tuner)
- Réduire la duplication de texte dans les appels LLM
- Préparer la mise en place d'un cache de contexte / prompt cache
"""

from typing import Dict

#
# Prompts d'enrichissement "métadonnées" (titre, résumé, etc.)
#

DEFAULT_ENRICHMENT_PROMPTS: Dict[str, str] = {
    # Contexte : appels client ↔ agent, en français
    # Prompts volontairement courts pour limiter le nombre de tokens
    "title": (
        "Appel entre un client et un agent. "
        "Génère un titre très court (max 10 mots), en français, sans explication ni commentaire."
    ),
    "summary": (
        "Appel entre un client et un agent. "
        "Génère un résumé concis (max 50 mots) en français, sans explication, en une à deux phrases."
    ),
    "satisfaction": (
        "Appel entre un client et un agent. "
        "Évalue la satisfaction du client sur une échelle de 1 à 10. "
        'Réponds UNIQUEMENT au format JSON : {"score": nombre}.'
    ),
    "bullet_points": (
        "Appel entre un client et un agent. "
        "Génère les points clés sous forme de puces. "
        'Réponds UNIQUEMENT au format JSON : {"points": ["point 1", "point 2", ...]}. '
        "Réponds en français."
    ),
}


#
# Prompts pour la correction segmentaire
#

SEGMENT_CORRECTION_BASE_PROMPT: str = (
    "Tu es un assistant qui CORRIGE et AMÉLIORE des transcriptions audio en français.\n"
    "RÈGLES STRICTES :\n"
    "1. Corriger UNIQUEMENT les erreurs d'orthographe et de grammaire.\n"
    "2. Améliorer UNIQUEMENT la ponctuation (points, virgules, majuscules).\n"
    "3. Améliorer UNIQUEMENT la structure (majuscules en début de phrase).\n"
    "4. CONSERVER EXACTEMENT le sens original : ne rien ajouter, ne rien inventer.\n"
    "5. Retourner UNIQUEMENT le texte corrigé, sans explication.\n"
    "6. La longueur du texte corrigé doit rester SIMILAIRE à l'original."
)

SEGMENT_CORRECTION_TASK_PROMPT: str = (
    "Corrige et améliore ce texte de transcription en conservant le sens original. "
    "Retourne UNIQUEMENT le texte corrigé :"
)


