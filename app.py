#!/usr/bin/env python3
"""
Agentic AI for Supply Chain Optimization — chatbot interface (Streamlit)

Run: streamlit run app.py
"""
import copy
import os

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.agents.chat_agent import build_context, handle_message, commentary
from src.models.supply_chain import AgentState
from src.utils.data_loader import load_file
from src.utils.llm import get_llm
from src.utils.report import baseline_markdown, summary_markdown
from src.utils import worst_case
from src.utils.sample_data import SAMPLE_PROBLEM
from src.workflow.supply_chain_workflow import get_supply_chain_workflow

load_dotenv()

st.set_page_config(page_title="Assistant planification Supply Chain", page_icon="🏭",
                   layout="wide", initial_sidebar_state="expanded")

WELCOME = ("Bonjour 👋 Parlez-moi naturellement : je choisis la meilleure façon de répondre. J'utilise OR-Tools quand "
           "il faut **calculer un planning**, et je réponds directement à partir de vos données sinon.\n\n"
           "1. Chargez vos données (ou l'exemple) dans la barre latérale.\n"
           "2. Par ex. *« montre le Gantt sans panne »*, *« panne de M2 pendant les deux premières heures »*, "
           "*« simule un retard de livraison d'un composant »*, *« quel est le pire cas d'une perturbation de 2 heures ? »* "
           "ou une question sur vos commandes. "
           "Je vous poserai les questions qui manquent.")


# --------------------------------------------------------------------------- state
def init_state():
    ss = st.session_state
    ss.setdefault("messages", [{"role": "assistant", "content": WELCOME}])
    ss.setdefault("pending", {})
    ss.setdefault("loaded_problem", None)
    ss.setdefault("data_label", "")
    ss.setdefault("uploaded_sig", None)
    ss.setdefault("last_facts", "")
    ss.setdefault("api_key", os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or "")
    if "llm" not in ss or ss.get("llm_key") != ss.api_key:
        ss.llm_key = ss.api_key
        ss.llm = get_llm(ss.api_key or None, temperature=0.2, max_tokens=4096)
        ss.workflow = get_supply_chain_workflow(ss.api_key or None)


def set_dataset(problem, label):
    st.session_state.loaded_problem = problem
    st.session_state.data_label = label
    st.session_state.pending = {}
    st.session_state.last_facts = ""
    comps = problem.get("component_availability", {})
    st.session_state.messages.append({"role": "assistant", "content": (
        f"✅ Données chargées (**{label}**) : {len(problem['jobs'])} commandes, "
        f"machines {', '.join(problem['machines'])}, "
        + (f"composants {', '.join(comps)}." if comps else "**aucun composant décrit** (retards de composants impossibles).")
    )})


# --------------------------------------------------------------------------- charts
def gantt_figure(result, problem, title, x_max, downtime=None):
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    job_color = {j.job_id: colors[i % len(colors)] for i, j in enumerate(problem.jobs)}
    seen = set()
    for machine_id, ops in result.schedule.items():
        for op in ops:
            jid = op["job_id"]
            fig.add_trace(go.Bar(
                x=[op["end"] - op["start"]], y=[machine_id], base=[op["start"]], orientation="h",
                name=jid, legendgroup=jid, showlegend=jid not in seen, marker_color=job_color[jid],
                marker_line_color="#444", marker_line_width=1,
                text=[op["operation_id"]], textposition="inside",
                hovertext=f"{jid} / {op['operation_id']}<br>début {op['start']} → fin {op['end']}"))
            seen.add(jid)
    for machine_id, ivs in (downtime or {}).items():
        for a, b in ivs:
            fig.add_trace(go.Bar(x=[b - a], y=[machine_id], base=[a], orientation="h",
                                 name="Indisponible", legendgroup="down", showlegend="down" not in seen,
                                 marker_color="rgba(220,0,0,0.45)", hovertext=f"{machine_id} indisponible [{a}, {b})"))
            seen.add("down")
    fig.update_layout(title=title, barmode="overlay", height=320, xaxis_title="Temps",
                      xaxis_range=[0, x_max + 1], margin=dict(l=10, r=10, t=40, b=10))
    return fig


def render_baseline(payload, key):
    ref, problem = payload["ref"], payload["problem"]
    c1, c2 = st.columns(2)
    c1.metric("Makespan", ref.makespan)
    c2.metric("Statut solveur", ref.status)
    if ref.status not in ("OPTIMAL", "FEASIBLE"):
        return
    st.plotly_chart(gantt_figure(ref, problem, "Planning de référence (sans perturbation)", ref.makespan),
                    key=f"{key}_g0", width="stretch")
    with st.expander("Commandes : fin et retard"):
        st.dataframe([{"Commande": j.job_id, "Livraison due": j.due_date,
                       "Fin": ref.job_completion_times.get(j.job_id),
                       "Retard": (max(0, ref.job_completion_times[j.job_id] - j.due_date)
                                  if j.due_date is not None and j.job_id in ref.job_completion_times else None)}
                      for j in problem.jobs], width="stretch")


def render_payload(payload, key):
    if payload.get("baseline"):
        return render_baseline(payload, key)
    ref, pert, impact, problem = payload["ref"], payload["pert"], payload["impact"], payload["problem"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Makespan avant", ref.makespan)
    c2.metric("Makespan après", pert.makespan, delta=None if impact is None else impact["makespan_pert"] - impact["makespan_ref"],
              delta_color="inverse")
    c3.metric("Statut solveur", f"{ref.status} / {pert.status}")
    if impact is None:
        return
    t1, t2, t3 = st.tabs(["Commandes", "Opérations (début / fin)", "Gantt avant / après"])
    with t1:
        st.dataframe([{
            "Commande": r["job_id"], "Livraison due": r["due_date"],
            "Fin avant": r["ref_completion"], "Fin après": r["pert_completion"], "Écart": r["delta"],
            "Retard avant": r["ref_lateness"], "Retard après": r["pert_lateness"],
            "Concernée directement": "oui" if r["concerned"] else "non",
            "Impactée": "oui" if r["impacted"] else "non"} for r in impact["jobs"]], width="stretch")
    with t2:
        st.dataframe([{
            "Commande": o["job_id"], "Opération": o["operation_id"], "Machine": o["machine_id"],
            "Début avant": o["ref_start"], "Fin avant": o["ref_end"],
            "Début après": o["pert_start"], "Fin après": o["pert_end"],
            "Décalée": "oui" if o["shifted"] else "non"} for o in impact["operations"]], width="stretch")
    with t3:
        x_max = max(impact["makespan_ref"], impact["makespan_pert"])
        st.plotly_chart(gantt_figure(ref, problem, "Avant la perturbation", x_max), key=f"{key}_g1", width="stretch")
        st.plotly_chart(gantt_figure(pert, problem, "Après la perturbation", x_max, problem.machine_availability),
                        key=f"{key}_g2", width="stretch")


def render_message(msg, key):
    st.markdown(msg["content"])
    if msg.get("payload"):
        render_payload(msg["payload"], key)
    if msg.get("commentary"):
        st.info(msg["commentary"])
    for w in msg.get("warnings", []):
        st.warning(w)


# --------------------------------------------------------------------------- turn
def process(prompt):
    ss = st.session_state
    lp = ss.loaded_problem
    ctx = build_context(lp) if lp else None
    turn = handle_message(prompt, ss.pending, ctx, llm=ss.llm, last_facts=ss.last_facts)
    ss.pending = turn["pending"]
    msg = {"role": "assistant", "content": turn["reply"]}
    if not turn["run"]:
        return msg

    run, analysis_md = turn["run"], ""
    if run["type"] == "worst_case":  # tool: try every disruption of this duration, then replay the worst one in detail
        try:
            res = worst_case.analyse(lp, run["duration"], run["scope"])
        except Exception as e:
            msg["content"] += f"\n\n❌ Erreur pendant l'analyse : {e}"
            return msg
        analysis_md = worst_case.markdown(res, run["duration"])
        if not res["rows"] or res["rows"][0]["makespan"] is None:
            msg["content"] += "\n\n" + analysis_md
            ss.last_facts = analysis_md
            return msg
        run = dict(res["rows"][0]["scenario"], notes=run.get("notes", []))
        turn = dict(turn, run=run)

    try:
        result = AgentState(**ss.workflow.invoke(
            AgentState(problem_description=prompt, loaded_problem=lp, scenario=turn["run"])))
    except Exception as e:
        msg["content"] += f"\n\n❌ Erreur pendant le calcul : {e}"
        return msg
    if result.error_messages:
        msg["content"] += "\n\n⚠️ " + "\n".join(result.error_messages)
        return msg

    problem, ref, pert = result.supply_chain_problem, result.reference_result, result.perturbed_result
    if turn["run"].get("type") == "baseline":  # tool "show_baseline": reference schedule only
        summary = baseline_markdown(problem, ref)
        msg["content"] += "\n\n" + summary
        msg["payload"] = {"baseline": True, "ref": ref, "problem": problem}
        ss.last_facts = summary + "\n\nOpérations (planning de référence): " + str(
            [(o["job_id"], o["operation_id"], m, o["start"], o["end"])
             for m, ops in ref.schedule.items() for o in ops])
        return msg
    summary = summary_markdown(problem, turn["run"], ref, pert, result.impact)
    if analysis_md:
        summary = analysis_md + "\n\n---\n**Détail du pire cas (avant / après) :**\n\n" + summary
    msg["content"] += "\n\n" + summary
    msg["payload"] = {"ref": ref, "pert": pert, "impact": result.impact, "problem": problem}
    msg["warnings"] = result.warnings
    if ss.llm is not None:
        msg["commentary"] = commentary(ss.llm, summary)
    ss.last_facts = summary + "\n\nOpérations (avant/après): " + str(
        [(o["job_id"], o["operation_id"], o["ref_start"], o["ref_end"], o["pert_start"], o["pert_end"])
         for o in (result.impact or {}).get("operations", [])])
    return msg


# --------------------------------------------------------------------------- UI
def sidebar():
    ss = st.session_state
    with st.sidebar:
        st.header("⚙️ Configuration")
        key = st.text_input("Clé API Groq (ou token Hugging Face)", type="password", value=ss.api_key,
                            help="Clé Groq (gsk_...) recommandée. Sans clé, le chatbot fonctionne quand même "
                                 "(règles), seules l'analyse rédigée et les questions libres sont désactivées.")
        if key != ss.api_key:
            ss.api_key = key
            if key:
                os.environ["GROQ_API_KEY" if not key.startswith("hf_") else "HF_TOKEN"] = key
            st.rerun()
        st.caption("🟢 LLM actif" if ss.llm is not None else "⚪ Mode sans LLM (règles + OR-Tools)")

        st.divider()
        st.subheader("📂 Données de production")
        up = st.file_uploader("Charger jobs / opérations / machines (JSON ou CSV)", type=["json", "csv"],
                              help="CSV : job_id, operation_id, machine_id, duration "
                                   "[, order_index, due_date, priority, components (ex. C1|C2)]")
        if up is not None and ss.uploaded_sig != (up.name, up.size):
            ss.uploaded_sig = (up.name, up.size)
            try:
                set_dataset(load_file(up.name, up.getvalue()), up.name)
            except Exception as e:
                st.error(f"Fichier invalide : {e}")
        if st.button("Charger l'exemple (fictif)"):
            set_dataset(copy.deepcopy(SAMPLE_PROBLEM), "exemple fictif")
            st.rerun()
        if ss.loaded_problem:
            st.success(f"{ss.data_label} : {len(ss.loaded_problem['jobs'])} commandes")
        else:
            st.info("Aucune donnée chargée.")

        st.divider()
        if st.button("🗑️ Nouvelle conversation"):
            ss.messages = [{"role": "assistant", "content": WELCOME}]
            ss.pending, ss.last_facts = {}, ""
            st.rerun()


def main():
    init_state()
    sidebar()
    st.title("🏭 Assistant de planification Supply Chain")

    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            render_message(m, f"m{i}")

    prompt = st.chat_input("Posez une question, demandez un Gantt ou décrivez une perturbation…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Je réfléchis…"):
                msg = process(prompt)
            idx = len(st.session_state.messages)
            render_message(msg, f"m{idx}")
        st.session_state.messages.append(msg)


main()