"""Jeu d'évaluation sémantique : ce que les modèles NLP doivent comprendre, sans contraintes parasites.

- GOLD : phrase d'intention -> catégories attendues dans le top 5 (précision@5 et taux de réussite).
- PARAPHRASES : groupes de formulations d'une même intention -> le top 5 doit rester stable (robustesse).

Les catégories sont celles du jeu de données (colonne « categories »).
"""

GOLD = [
    ("ballon en équipe", "J'adore taper dans un ballon avec mes copains sur un terrain.",
     {"jeu de ballon", "sport collectif"}),
    ("combat", "Je veux me défouler en frappant et en esquivant, j'aime les sports de combat.",
     {"sport de combat", "combat de percussion", "combat de préhension"}),
    ("détente et souplesse", "J'ai envie de me détendre, de respirer et d'étirer doucement mon corps.",
     {"mobilité", "relaxation"}),
    ("course rapide", "Je veux courir dehors et améliorer ma vitesse sur de courtes distances.",
     {"course", "athlétisme"}),
    ("neige", "Je rêve de glisser sur la neige en montagne en hiver.",
     {"glisse sur neige", "terrain naturel"}),
    ("natation", "J'aime nager et être dans l'eau pour travailler tout le corps.",
     {"nage", "sport aquatique"}),
    ("vélo", "Je veux pédaler longtemps sur une route ou un chemin, en plein air.",
     {"cyclisme"}),
    ("raquette", "Je veux un sport avec une raquette pour jouer des échanges rapides avec un partenaire.",
     {"sport de raquette"}),
    ("marche nature", "Je cherche une activité de marche tranquille dans la nature pour réfléchir.",
     {"marche", "terrain naturel"}),
    ("escalade", "J'aimerais grimper sur des rochers et des prises en hauteur.",
     {"escalade"}),
    ("précision", "Je veux viser une cible avec précision et concentration, dans le calme.",
     {"sport de cible", "précision"}),
    ("danse", "Je veux danser et bouger au rythme de la musique avec une chorégraphie.",
     {"danse", "chorégraphie"}),
    ("force en salle", "Je veux faire de la musculation et prendre de la force dans une salle.",
     {"sport en salle", "force"}),
    ("sport d'équipe", "J'aime les sports d'équipe où il faut passer le ballon et marquer.",
     {"sport collectif", "jeu de ballon"}),
]

PARAPHRASES = [
    ("détente", [
        "Je veux me détendre et étirer mon corps doucement.",
        "J'ai besoin de relâcher mes muscles et de gagner en souplesse calmement.",
        "Je cherche une pratique apaisante pour assouplir mon corps.",
    ]),
    ("course", [
        "Je veux courir dehors vite.",
        "J'ai envie de faire du sprint en extérieur.",
        "Je veux pratiquer la course à pied rapide au grand air.",
    ]),
    ("natation", [
        "J'adore nager dans l'eau.",
        "J'aime la natation et les bassins.",
        "Je veux pratiquer une activité aquatique en nageant.",
    ]),
    ("ballon", [
        "Je veux taper dans un ballon avec des amis.",
        "J'aime jouer au foot avec mes copains.",
        "Je veux un sport de ballon en équipe.",
    ]),
    ("combat", [
        "Je veux m'entraîner à la boxe et aux combats.",
        "J'aime les sports de combat et frapper.",
        "Je veux faire des arts martiaux et me battre en pratique.",
    ]),
]
