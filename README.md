# vocalyx-enrichment

Worker Celery pour l'enrichissement des transcriptions avec LLM (CPU only).

## Description

Ce module enrichit les transcriptions avec des informations générées par un LLM :
- **Titre** : Titre court et accrocheur (max 10 mots)
- **Résumé** : Résumé concis de moins de 100 mots
- **Score de satisfaction** : Score de 1 à 10 avec justification
- **Bullet points** : Points clés sous forme de puces

## Configuration

Le module est configuré via `config.ini` ou des variables d'environnement :

- `INSTANCE_NAME` : Nom de l'instance du worker
- `VOCALYX_API_URL` : URL de l'API centrale
- `CELERY_BROKER_URL` : URL du broker Redis
- `LLM_DEVICE` : Device à utiliser (cpu)
- `LLM_MODEL` : Modèle LLM à utiliser (stocké en base de données par transcription)
- `INTERNAL_API_KEY` : Clé d'authentification interne

## Prompts personnalisables

Les prompts peuvent être personnalisés lors de la création de transcription via l'API ou l'interface web.

## Format de sortie

Les données d'enrichissement sont stockées en JSON dans la base de données :

```json
{
  "title": "Titre généré",
  "summary": "Résumé de moins de 100 mots...",
  "satisfaction_score": 8,
  "satisfaction_justification": "Justification du score",
  "bullet_points": ["Point 1", "Point 2", "..."],
}
```

