"""Interface Streamlit des bonus IA : bouton d'explication (bonus A) et espace de conversation (bonus B).

Ce module ne calcule rien : il affiche les résultats de `bonus_llm` et `bonus_chatbot`, en distinguant clairement :
    - les scores calculés par le moteur NLP ;
    - les explications générées par l'IA (étiquetées comme telles) ;
    - les préférences ajoutées pendant la conversation ;
    - le classement mis à jour.
"""
import streamlit as st

import bonus_chatbot as bc
import bonus_llm as bl
from engine import BLOCK_LABELS, BLOCKS
from llm_config import LLMSettings

ASSISTANT_INTRO = ("Bonjour ! Je peux modifier votre profil (préférences, contraintes, budget, disponibilité) et relancer "
                   "le moteur, ou répondre à vos questions sur les sports recommandés. Exemple : « Finalement, je n'aime "
                   "pas les sports aquatiques et je préfère les activités individuelles en extérieur. »")


def _budget(settings: LLMSettings) -> bl.CallBudget:
    if "llm_budget" not in st.session_state:
        st.session_state["llm_budget"] = bl.CallBudget(settings.max_calls_per_session)
    return st.session_state["llm_budget"]


def render_explain_button(out: dict, res, settings: LLMSettings) -> None:
    """Bonus A : bouton « Expliquer cette recommandation avec l'IA » sous une carte de sport."""
    cache = st.session_state.setdefault("ia_cache", {})
    key = bl.cache_key(out, res.name, settings)
    clicked = st.button("✨ Expliquer avec l'IA", key=f"ia_btn_{res.sport_id}",
                        help="Un modèle de langage rédige une explication à partir des scores déjà calculés.")
    if clicked and key not in cache:
        with st.spinner("Rédaction de l'explication…"):
            cache[key] = bl.explain_sport(out, res.name, settings, budget=_budget(settings))
    if key not in cache:
        return
    ex = cache[key]
    if ex["source"] == "llm":
        with st.container(border=True):
            st.markdown("**✨ Explication générée par l'IA**")
            st.caption(f"Modèle : {settings.describe()} · texte rédigé à partir des scores calculés par le moteur NLP, "
                       "vérifié avant affichage. Il ne modifie pas le classement.")
            st.write(ex["text"])
    else:
        st.warning(f"Explication IA indisponible : {ex['reason']}. Voici l'explication classique de l'application.")
        for line in ex["lines"]:
            st.markdown(f"{'✅' if line['ok'] else '⚠️'} **{line['block']}** — {line['text']}")


def _changes_from_history(chat: list[dict]) -> list[str]:
    out = []
    for m in chat:
        if m.get("kind") == "maj":
            out.extend(m.get("changes", []))
    return out


def render_chat(out: dict, profile, engine, weights: dict, settings: LLMSettings) -> None:
    """Bonus B : espace de conversation. Une modification met à jour le profil et relance le moteur."""
    chat = st.session_state.setdefault("chat", [])
    history = st.session_state.setdefault("profile_history", [])

    st.caption("Votre message est lu par des règles et, si l'IA est configurée, par un modèle de langage qui ne renvoie "
               "que des champs de profil validés. Ne saisissez pas d'informations d'identité.")
    if not chat:
        with st.chat_message("assistant"):
            st.markdown(ASSISTANT_INTRO)
    for m in chat:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m.get("tag"):
                st.caption(m["tag"])

    cols = st.columns([3, 1])
    if history and cols[1].button("↩️ Annuler la dernière modification"):
        st.session_state["profile"] = history.pop()
        chat.append({"role": "assistant", "text": "Dernière modification annulée : le classement est recalculé.",
                     "tag": "Annulation", "kind": "aide"})
        st.rerun()

    msg = st.chat_input("Ex. : je m'entraîne 4 fois par semaine, mon budget est de 15 € par séance…")
    if not msg:
        return
    res = bc.process_message(msg, profile, out, engine, weights, settings, budget=_budget(settings))
    tag = f"Source : {res['source']}" + (f" · {res['note']}" if res.get("note") else "")
    chat.append({"role": "user", "text": msg})
    chat.append({"role": "assistant", "text": res["text"], "tag": tag, "kind": res["kind"],
                 "changes": res.get("changes", [])})
    if res["kind"] == "maj":
        history.append(profile)
        st.session_state["profile"] = res["profile"]
        st.rerun()  # relance toute la page : le classement affiché est celui du nouveau profil


def render_profile_panel() -> None:
    """Préférences et contraintes ajoutées pendant la conversation, séparées du questionnaire initial."""
    changes = _changes_from_history(st.session_state.get("chat", []))
    if not changes:
        return
    with st.expander(f"🧩 Ajouté pendant la conversation ({len(changes)})", expanded=True):
        for c in changes:
            st.write("• " + c)
