"""Scénarios de test partagés par le panneau Admin et le benchmark des modèles.

Chaque scénario : (nom, texte utilisateur, fonction de vérification(o, df) -> bool).
"""
from engine import WATER_TAGS  # noqa: F401  (utilisé par les vérifications)


def _top(o, df):
    """Lignes du jeu de données correspondant au top 5 retourné."""
    return df[df.nom_sport.isin([r.name for r in o["results"]])]


PERSONAS = [
    ("Genou fragile : pas de sauts ni d'impacts fréquents",
     "J'ai mal au genou, je dois éviter les sauts et les chocs. Je veux me détendre et travailler mon équilibre.",
     lambda o, df: all(df.loc[df.nom_sport == r.name, "impacts_1_3"].iloc[0] < 3 and "saut" not in df.loc[df.nom_sport == r.name, "categories_l"].iloc[0] for r in o["results"])),
    ("Ne sait pas nager : aucun sport aquatique",
     "Je ne sais pas nager. Je veux améliorer mon endurance, plutôt en extérieur.",
     lambda o, df: not any(set(df.loc[df.nom_sport == r.name, "categories_l"].iloc[0]) & WATER_TAGS for r in o["results"])),
    ("Pas de contact : aucun contact réglementé",
     "Je ne veux surtout pas de sport de contact, je cherche à développer ma force et mon cardio.",
     lambda o, df: all(df.loc[df.nom_sport == r.name, "contact"].iloc[0] != "réglementé" for r in o["results"])),
    ("Budget 10 € : respecte le plafond",
     "Mon budget est de 10 euros par séance maximum. Je veux me remettre à l'endurance.",
     lambda o, df: all(df.loc[df.nom_sport == r.name, "budget_seance_min_eur"].iloc[0] <= 10 for r in o["results"])),
    ("Exemple de l'énoncé : un sport doux d'endurance dans le top 5",
     "Je suis débutant en sport, je me fatigue rapidement lorsque je cours, mais je peux marcher assez longtemps. "
     "Je souhaite améliorer mon endurance, renforcer mon dos et pratiquer une activité qui ne sollicite pas trop mes articulations. "
     "Je préfère les activités individuelles et je suis disponible trois fois par semaine.",
     lambda o, df: bool(((_top(o, df).endurance_1_5 >= 4) & (_top(o, df).impacts_1_3 == 1)).any())),
    ("Passionné de combat : au moins 3 sports de combat",
     "Je suis confirmé, j'adore les sports de combat et la boxe, je veux de l'intensité et progresser techniquement.",
     lambda o, df: sum(bool({"sport de combat", "combat de percussion", "combat de préhension"} & set(df.loc[df.nom_sport == r.name, "categories_l"].iloc[0])) for r in o["results"]) >= 3),
    ("Amateur d'équipe : au moins 3 sports collectifs",
     "J'aime les sports collectifs, jouer en équipe avec des amis, courir et me défouler.",
     lambda o, df: sum("sport collectif" in df.loc[df.nom_sport == r.name, "categories_l"].iloc[0] for r in o["results"]) >= 3),
    ("Épaule blessée, tennis pratiqué : doux, intérieur, sans épaule ni raquette",
     "Je pratique la salle de sport et le tennis mais je me suis blessé à l'épaule donc j'aimerais trouver un sport plus calme et plus doux qui pourrait me soulager l'épaule pendant ma réhabilitation. Je préfère un sport d'intérieur qui ne pose pas de problème avec la météo.",
     lambda o, df: all("sport de raquette" not in df.loc[df.nom_sport == r.name, "categories_l"].iloc[0]
                       and df.loc[df.nom_sport == r.name, "intensite_loisir_1_5"].iloc[0] <= 3
                       and "intérieur" in df.loc[df.nom_sport == r.name, "environnement_l"].iloc[0] for r in o["results"])
                  and sum("épaules" in df.loc[df.nom_sport == r.name, "zones_corps_sollicitees_l"].iloc[0] for r in o["results"]) <= 1),
    ("Tendinite au coude : pas de raquette, pas d'avant-bras sollicité",
     "J'ai une tendinite au coude, j'ai mal à l'avant-bras. Je veux garder la forme et travailler mon endurance en douceur.",
     lambda o, df: all("avant-bras" not in df.loc[df.nom_sport == r.name, "zones_corps_sollicitees_l"].iloc[0] for r in o["results"])),
    ("Vertiges : au plus un sport très exigeant en équilibre dans le top 5",
     "J'ai des vertiges et je manque d'équilibre quand je marche. Je veux bouger en douceur et me détendre.",
     lambda o, df: int((_top(o, df).equilibre_1_5 >= 4).sum()) <= 1),
    ("Détente et souplesse : au moins 2 pratiques de mobilité ou de bien-être",
     "Je suis débutant, je veux me détendre et gagner en souplesse. Je préfère pratiquer en intérieur, plutôt seul.",
     lambda o, df: int(_top(o, df).categories_l.apply(lambda c: bool({"mobilité", "relaxation"} & set(c))).sum()) >= 2),
    ("Salle et force : au moins 2 sports de salle orientés force",
     "Je m'entraîne en salle de sport depuis plusieurs années. Je veux prendre de la force et de la puissance, je m'entraîne seul.",
     lambda o, df: int(_top(o, df).apply(lambda r: "sport en salle" in r.categories_l and r.force_1_5 >= 4, axis=1).sum()) >= 2),
    ("Dos en douceur : au moins 2 sports qui travaillent le dos",
     "Je veux renforcer mon dos en douceur, sans trop d'impacts.",
     lambda o, df: int(_top(o, df).zones_corps_sollicitees_l.apply(lambda z: "dos" in z).sum()) >= 2),
    ("Équipe sans contact : aucun sport à contact réglementé, et au moins un sport collectif",
     "Je veux jouer en équipe avec des amis, mais sans aucun contact physique.",
     lambda o, df: bool((_top(o, df).contact != "réglementé").all()) and bool(_top(o, df).categories_l.apply(lambda c: "sport collectif" in c).any())),
    ("Pluie et intérieur : tous les sports proposés se pratiquent en intérieur",
     "Je veux faire du cardio en intérieur, car il pleut souvent où j'habite. Je suis régulier.",
     lambda o, df: bool(_top(o, df).environnement_l.apply(lambda e: "intérieur" in e).all())),
    ("Budget nul : seulement des activités sans coût de séance",
     "Mon budget est de 0 euro, je veux marcher et me promener dehors.",
     lambda o, df: bool((_top(o, df).budget_seance_min_eur <= 0).all())),
]
