"""Extraction d'indices (objectifs, contraintes, préférences, condition) dans le texte libre.

Approche à base de lexiques sur texte sans accents, avec détection de la négation
(« pas de contact », « je n'aime pas l'eau ») dans une fenêtre de caractères précédant l'expression.
"""
import re
from dataclasses import dataclass, field

from preprocess import fold, split_sentences

NEG = r"(?:\bpas\b|\bsans\b|\bjamais\b|\bni\b|\baucun\w*|\bevit\w+|\bn'aime\b|\bdeteste\w*)"

# --- objectifs : étiquette (dataset / capacité) -> regex ------------------------------------
GOAL_LEX = {
    "endurance": r"endurance|cardio|souffle|tenir plus longtemps|resistance a l'effort",
    "force": r"\bforce\b|muscl|renforc|tonifi|tonicite|gainage",
    "souplesse": r"souplesse|flexib|etirement|mobilite|assouplir|\bsouple",
    "équilibre": r"equilibre|stabilite|proprioception",
    "coordination": r"coordination|motricite|dexterite",
    "détente": r"detente|relax|bien-etre|bien etre|stress|decompress|apais|se vider la tete|zen",
    "vitesse": r"vitesse|rapidite|sprint",
    "puissance": r"puissance|explosiv",
    "précision": r"precision|concentration|adresse",
    "jeu collectif": r"jeu collectif|esprit d'equipe|cohesion|convivial|se faire des amis|rencontrer|socialis",
    "plein air": r"plein air|nature|dehors|grand air",
    "agilité": r"agilite|reactivite|reflexes?",
    "progression technique": r"progresser|apprendre|technique|maitriser",
    "expression corporelle": r"expression|danse|creativ|rythme",
    "gestion de l'effort": r"gestion de l'effort|regularite|rythme regulier",
}
GOAL_TRIGGER = (r"souhait|voudrais|veux|aimerais|objectif|ameliorer|amelioration|developper|travailler|renforc"
                r"|gagner|progress|augmenter|retrouver|decouvrir|besoin|envie|cherche")

ZONE_LEX = {
    "dos": r"\bdos\b|lombaire|dorsal|colonne",
    "épaules": r"epaule",
    "bras": r"\bbras\b|biceps|triceps",
    "abdominaux": r"abdo|ventre|sangle abdominale|\bcore\b",
    "cuisses": r"cuisse|quadriceps|jambes?",
    "hanches et fessiers": r"hanche|fessier|bassin",
    "mollets": r"mollet",
    "pieds et chevilles": r"cheville|\bpieds?\b",
    "mains et poignets": r"poignet|\bmains?\b",
}
ZONE_GOAL_VERB = r"(?:renforc|muscl|tonifi|travaill|fortifi|developp|gainer|affin)\w*"
ZONE_PAIN = (r"(?:mal (?:au|aux|a)|douleurs?|blessure|blesse|fragile|faiblesse|probleme(?:s)? (?:de|au|aux)"
             r"|operee?|tendinite|lumbago|hernie|entorse)")

PREF_LEX = {
    "format:individuel": r"individuel|seul\b|en solo|tout seul|toute seule|a mon rythme",
    "format:équipe": r"collectif|en equipe|equipe\b|avec des amis",
    "format:duo": r"en duo|a deux|avec un partenaire|face a face",
    "env:extérieur": r"exterieur|dehors|plein air|en nature|grand air",
    "env:intérieur": r"interieur|en salle|\bsalle\b|gymnase|a l'abri",
    "fam:ballon": r"ballon|football|\bfoot\b|basket|volley|hand\b|rugby",
    "fam:raquette": r"raquette|tennis|badminton|ping|padel|squash",
    "fam:combat": r"combat|boxe|judo|karate|arts? martiaux",
    "fam:eau": r"\beau\b|natation|nager|piscine|aquatique|nautique",
    "fam:glisse": r"glisse|ski|surf|snowboard|skate|roller|patin",
    "fam:course": r"course|courir|running|footing|trail",
    "fam:cyclisme": r"velo|cyclisme|vtt|bicyclette",
    "fam:gym-danse": r"danse|gymnastique|acrobat|trampoline|fitness",
    "fam:cible": r"cible|precision|golf|petanque|arc\b|bowling",
    "fam:marche": r"randonnee|marche|balade|trek",
    "fam:escalade": r"escalade|grimp|bloc\b",
}

PREF_CUE = r"prefer|aime|adore|plutot|envie|attir|passionn|pratiquer|deteste|interess|collectif|individuel"

CONSTRAINT_LEX = {
    "no_contact": r"(?:pas|sans|aucun|evit\w+)\W+(?:\w+\W+){0,3}(?:contact|combat|bagarre)",
    "joint_sensitive": r"articulation|genou|arthrose|tendin|menisque|\bchocs?\b|\bimpacts?\b",
    "no_jump": r"(?:pas|sans|evit\w+)\W+(?:\w+\W+){0,3}sauts?|ne pas sauter",
    "no_water": r"(?:ne sais pas nager|sais pas nager|peur de l'eau|aquaphobe|pas d'eau|sans eau)",
    "no_outdoor": r"(?:pas|sans|evit\w+)\W+(?:\w+\W+){0,2}(?:exterieur|dehors|froid|pluie)",
    "no_indoor": r"(?:pas|sans|evit\w+)\W+(?:\w+\W+){0,2}(?:salle|interieur|gymnase)",
}

CONDITION_LEX = [
    # (capacité, regex, delta, libellé)
    ("endurance", r"(?:me )?fatigu\w+ (?:vite|rapidement|facilement)|essouffl|souffle court|manque d'endurance", -1, "fatigue rapide / essoufflement"),
    ("endurance", r"marche\w* (?:assez |tres )?longtemps|bonne endurance|bon souffle|tiens? longtemps", +1, "bonne endurance à la marche"),
    ("force", r"manque de force|peu de force|musculature faible", -1, "force faible déclarée"),
    ("force", r"musculature solide|bonne force|porte facilement", +1, "bonne force déclarée"),
    ("souplesse", r"raide|manque de souplesse|peu souple|courbatu", -1, "raideur déclarée"),
    ("souplesse", r"tres souple|bonne souplesse|bonne mobilite", +1, "bonne souplesse déclarée"),
]
LEVEL_LEX = [
    ("débutant", r"debutant|jamais (?:fait|pratique)|sedentaire|je reprends|pas sportif|novice"),
    ("confirmé", r"confirme|experimente|competition|haut niveau|depuis (?:des )?annees"),
]


@dataclass
class TextSignals:
    goals: set = field(default_factory=set)
    goal_zones: set = field(default_factory=set)
    spare_zones: set = field(default_factory=set)
    prefs: set = field(default_factory=set)
    disliked: set = field(default_factory=set)
    constraints: set = field(default_factory=set)
    condition_adj: dict = field(default_factory=dict)
    condition_notes: list = field(default_factory=list)
    level_hint: str | None = None
    budget: int | None = None
    sentences: list = field(default_factory=list)


def _negated(folded: str, start: int, window: int = 28) -> bool:
    return re.search(NEG, folded[max(0, start - window):start]) is not None


def extract_signals(text: str) -> TextSignals:
    sig = TextSignals()
    t = fold(text)
    if not t:
        return sig
    sig.sentences = split_sentences(text)

    for sent in (fold(s) for s in sig.sentences):
        has_goal_verb = re.search(GOAL_TRIGGER, sent) is not None
        for zone, rx in ZONE_LEX.items():
            for m in re.finditer(rx, sent):
                before = sent[max(0, m.start() - 45):m.start()]
                if re.search(ZONE_PAIN, before) or re.search(ZONE_PAIN, sent[m.end():m.end() + 25]):
                    sig.spare_zones.add(zone)
                elif re.search(ZONE_GOAL_VERB, before) or has_goal_verb:
                    sig.goal_zones.add(zone)
        if has_goal_verb:
            for goal, rx in GOAL_LEX.items():
                m = re.search(rx, sent)
                if m and not _negated(sent, m.start()):
                    sig.goals.add(goal)

    # préférences : uniquement dans les phrases qui expriment un goût (évite « je peux marcher longtemps » = préférence)
    pref_text = " . ".join(f for f in (fold(s) for s in sig.sentences) if re.search(PREF_CUE, f))
    for key, rx in PREF_LEX.items():
        for m in re.finditer(rx, pref_text):
            (sig.disliked if _negated(pref_text, m.start()) else sig.prefs).add(key)
    for key, rx in CONSTRAINT_LEX.items():
        if re.search(rx, t):
            sig.constraints.add(key)
    for cap, rx, delta, label in CONDITION_LEX:
        if re.search(rx, t):
            sig.condition_adj[cap] = sig.condition_adj.get(cap, 0) + delta
            sig.condition_notes.append(label)
    for lvl, rx in LEVEL_LEX:
        if re.search(rx, t):
            sig.level_hint = lvl
    m = re.search(r"(\d{1,3})\s*(?:euros?|€)", t)
    if m and re.search(r"budget|maximum|max|pas plus|au plus|jusqu'a|moins de|par seance|la seance", t):
        sig.budget = int(m.group(1))
    sig.prefs -= sig.disliked
    if "no_contact" in sig.constraints:
        sig.prefs.discard("fam:combat")
    return sig


# --- routage des phrases vers les blocs sémantiques ---------------------------------------
BLOCK_CUES = {
    "condition": r"debutant|niveau|fatigu|essouffl|endurance|marche|cours|force|souple|raide|sportif|sedentaire|depuis|habitude|pratique|experience|forme|capacite",
    "objectifs": GOAL_TRIGGER,
    "preferences": r"prefere|aime|adore|plutot|individuel|collectif|equipe|exterieur|interieur|seul|groupe|ballon|eau|nature|plaisir|ludique",
    "contraintes": r"mal |douleur|blessure|eviter|pas trop|sans |limit|articulation|genou|probleme|budget|euros|disponible|fois par semaine|seances?|minutes|heure|peur|contact|fragile",
}


def route_sentences(text: str) -> dict[str, list[str]]:
    routed = {b: [] for b in BLOCK_CUES}
    for s in split_sentences(text):
        f = fold(s)
        hit = False
        for block, rx in BLOCK_CUES.items():
            if re.search(rx, f):
                routed[block].append(s)
                hit = True
        if not hit:
            routed["objectifs"].append(s)
            routed["condition"].append(s)
    return routed
