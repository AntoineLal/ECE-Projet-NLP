"""Interface Streamlit des bonus IA : bouton d'explication (bonus A) et espace de conversation (bonus B).

Ce module ne calcule rien : il affiche les résultats de `bonus_llm` et `bonus_chatbot`, en séparant clairement :
    - les scores calculés par le moteur NLP ;
    - les explications générées par l'IA (étiquetées comme telles) ;
    - le profil actualisé (objectifs, préférences souples, contraintes strictes) ;
    - les changements depuis la dernière réponse, et le nouveau top 3 avec ses raisons.
"""
import streamlit as st

import bonus_chatbot as bc
import bonus_llm as bl
from engine import BLOCK_LABELS, BLOCKS
from llm_config import LLMSettings

ASSISTANT_INTRO = ("Bonjour ! Je peux modifier votre profil et relancer le moteur NLP, ou répondre à vos questions sur les "
                   "sports. Exemples : « Je ne veux aucun sport aquatique. », « Je veux uniquement des sports individuels. », "
                   "« Finalement, j'accepte aussi les sports collectifs. »")


def _budget(settings: LLMSettings) -> bl.CallBudget:
    if "llm_budget" not in st.session_state:
        st.session_state["llm_budget"] = bl.CallBudget(settings.max_calls_per_session)
    return st.session_state["llm_budget"]


# ------------------------------------------------------------------ bonus A
def render_explain_button(out: dict, res, settings: LLMSettings) -> None:
    """Bouton « Expliquer cette recommandation avec l'IA » sous une carte de sport."""
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
        st.info(f"Explication construite à partir des résultats du moteur NLP (IA non utilisée : {ex['reason']}).")
        st.write(ex["text"])
        with st.expander("Détail des critères calculés par le moteur"):
            for line in ex["lines"]:
                st.markdown(f"{'✅' if line['ok'] else '⚠️'} **{line['block']}** — {line['text']}")


# ------------------------------------------------------------------ bonus B : affichage d'une réponse
def _render_profile(view: dict) -> None:
    st.markdown("**Profil actualisé**")
    st.write("• **Objectifs retenus** : " + (", ".join(view["objectifs"]) or "aucun"))
    st.write("• **Préférences (souples, elles favorisent sans exclure)** : " + (" ; ".join(view["preferences"]) or "aucune"))
    st.write("• **Contraintes strictes (jamais contournées)** : " + (" ; ".join(view["contraintes_strictes"]) or "aucune"))


def _render_top(card: dict) -> None:
    st.markdown("**Nouvelles recommandations (moteur NLP)**")
    if card["n_kept"] == 0:
        st.error("Aucun sport du dataset ne respecte toutes vos contraintes. Aucune n'a été relâchée automatiquement.")
        if card["suggestions"]:
            st.markdown("Si vous acceptez de modifier une contrainte, voici ce que cela permettrait (sans l'appliquer) :")
            for s in card["suggestions"]:
                st.write("• " + s)
        return
    for row in card["top"]:
        with st.container(border=True):
            st.markdown(f"**{row['rang']}. {row['sport']}** — score global **{row['score']} %**")
            st.caption("Blocs : " + " · ".join(f"{k} {v} %" for k, v in row["blocs"].items() if v is not None))
            st.write("Critères de correspondance : " + (", ".join(row["criteres"]) or "aucun critère marqué"))
    cmp = card["comparison"]
    if cmp["identique"]:
        st.caption("Le top 3 est identique à la réponse précédente.")
    for e in cmp["entrees"]:
        st.write(f"➕ **Entre dans le top 3 : {e['sport']}** — " + " ; ".join(e["raisons"]))
    for s in cmp["sorties"]:
        st.write(f"➖ **Sort du top 3 : {s['sport']}** — " + " ; ".join(s["raisons"]))


def _render_card(card: dict) -> None:
    """Affiche une mise à jour complète : profil, changements, nouvelles recommandations."""
    _render_profile(card["view"])
    if card["changes"]:
        st.markdown("**Changements depuis la dernière réponse**")
        for c in card["changes"]:
            st.write("• " + c)
    if card["contradictions"]:
        st.markdown("**Contradictions résolues** (la dernière information prime)")
        for c in card["contradictions"]:
            st.write("• " + c)
    if card["rejected"]:
        st.caption("Informations non retenues (hors du contrat, valeur invalide) : " + " ; ".join(card["rejected"][:4]))
    _render_top(card)


# ------------------------------------------------------------------ bonus B : conversation
def render_chat(out: dict, profile, engine, weights: dict, settings: LLMSettings) -> None:
    """Espace de conversation. Une modification valide met le profil à jour et relance le moteur."""
    chat = st.session_state.setdefault("chat", [])
    st.caption("Votre message est lu par des règles et, si l'IA est configurée, par Gemini qui ne renvoie qu'un contrat "
               "JSON validé. Ne saisissez pas d'informations d'identité.")
    if not chat:
        with st.chat_message("assistant"):
            st.markdown(ASSISTANT_INTRO)
    for m in chat:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m.get("tag"):
                st.caption(m["tag"])
            if m.get("card"):
                _render_card(m["card"])
            for q in m.get("precisions", []):
                st.write("❓ " + q)

    if st.button("↩️ Réinitialiser les modifications de la conversation",
                 help="Revient au profil du questionnaire. Le questionnaire lui-même n'est pas effacé."):
        base = st.session_state.get("base_profile")
        if base is not None:
            st.session_state["profile"] = bc.reset_profile(base)
            chat.append({"role": "assistant", "text": "Modifications de la conversation réinitialisées : le profil est revenu "
                                                       "à celui du questionnaire.", "tag": "Réinitialisation"})
            st.rerun()

    msg = st.chat_input("Ex. : je ne veux aucun sport aquatique…")
    if not msg:
        return
    res = bc.process_message(msg, profile, out, engine, weights, settings, budget=_budget(settings))
    tag = f"Lecture : {res['source']}" + (f" · {res['note']}" if res.get("note") else "")
    chat.append({"role": "user", "text": msg})
    if res["kind"] == "maj":
        card = {"view": res["view"], "changes": res["changes"], "contradictions": res["contradictions"],
                "rejected": res["rejected"], "precisions": res["precisions"], "n_kept": res["n_kept"],
                "suggestions": res["suggestions"], "comparison": res["comparison"],
                "top": bc.top_summary(res["out"])}
        chat.append({"role": "assistant", "text": "Profil mis à jour et classement recalculé.", "tag": tag,
                     "card": card, "kind": "maj"})
        st.session_state["profile"] = res["profile"]
        st.rerun()  # la page est relancée : le top 5 affiché correspond au nouveau profil
    chat.append({"role": "assistant", "text": res["text"], "tag": tag, "kind": res["kind"],
                 "precisions": res.get("precisions", [])})
    st.rerun()
