from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clean"
OUT = ROOT / "outputs"

st.set_page_config(page_title="Instamart Intelligence", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")

# Lightweight helpers keep the dashboard readable and tolerate missing generated outputs.
def read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()

def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default

def inr(value):
    return f"₹{value:,.0f}"

def section_note(text):
    st.caption(text)

orders = read_csv(DATA / "orders.csv")
customers = read_csv(DATA / "customers.csv")
products = read_csv(DATA / "products.csv")
segments = read_csv(OUT / "customer_segments.csv")
segment_profile = read_csv(OUT / "segment_profile.csv")
retention = read_csv(OUT / "retention_list.csv")
tickets = read_csv(OUT / "triage_results.csv")
part_a = read_json(OUT / "part_a_summary.json")
part_b = read_json(OUT / "part_b_summary.json")
part_c = read_json(OUT / "part_c_summary.json")
part_d = read_json(OUT / "part_d_summary.json")

if not orders.empty:
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce").fillna(0)
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders["month"] = orders["order_ts"].dt.to_period("M").astype(str)

# Sidebar filters
st.sidebar.title("🛒 Instamart Copilot")
st.sidebar.write("Customer intelligence & support dashboard")
if not orders.empty:
    city_options = ["All cities"] + sorted(customers["city"].dropna().unique().tolist()) if "city" in customers else ["All cities"]
    selected_city = st.sidebar.selectbox("Customer city", city_options)
    status_options = ["All statuses"] + sorted(orders["status"].dropna().unique().tolist())
    selected_status = st.sidebar.selectbox("Order status", status_options)
    date_min = orders["order_ts"].min().date() if orders["order_ts"].notna().any() else None
    date_max = orders["order_ts"].max().date() if orders["order_ts"].notna().any() else None
    date_range = st.sidebar.date_input("Order date range", value=(date_min, date_max) if date_min and date_max else ())
else:
    selected_city, selected_status, date_range = "All cities", "All statuses", ()

filtered_orders = orders.copy()
if not filtered_orders.empty:
    if selected_status != "All statuses":
        filtered_orders = filtered_orders[filtered_orders["status"] == selected_status]
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        filtered_orders = filtered_orders[(filtered_orders["order_ts"] >= start_date) & (filtered_orders["order_ts"] < end_date)]
    if selected_city != "All cities" and not customers.empty:
        city_customer_ids = set(customers.loc[customers["city"] == selected_city, "customer_id"])
        filtered_orders = filtered_orders[filtered_orders["customer_id"].isin(city_customer_ids)]

st.title("Instamart Customer Intelligence & Support Copilot")
st.write("Explore order performance, customer segments, churn risk, support tickets, and policy-agent evaluation—all from the project's generated data.")
st.info("This dashboard uses the synthetic project dataset and saved model outputs. It does not connect to live Instamart systems or issue real refunds.")

completed_orders = filtered_orders[filtered_orders["status"].astype(str).str.lower() == "completed"] if not filtered_orders.empty else pd.DataFrame()
revenue = float(completed_orders["order_amount"].sum()) if not completed_orders.empty else 0.0
order_count = len(filtered_orders)
customer_count = customers[customers["city"] == selected_city].shape[0] if selected_city != "All cities" and not customers.empty else len(customers)
completion_rate = (len(completed_orders) / order_count * 100) if order_count else 0
lapse_rate = float(part_b.get("lapsed_rate", 0)) * 100
few_shot_accuracy = float(part_c.get("few_shot", {}).get("accuracy", 0)) * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Completed revenue", inr(revenue), help="Revenue from completed orders in the selected filters")
k2.metric("Orders in view", f"{order_count:,}")
k3.metric("Customers", f"{customer_count:,}")
k4.metric("Completion rate", f"{completion_rate:.1f}%")
k5.metric("Lapse rate", f"{lapse_rate:.1f}%", help="Modeling cohort lapse rate; not recalculated by the sidebar filters")

page = st.tabs(["Overview", "Orders & Sales", "Customers & Churn", "Support Tickets", "AI Copilot & Policies", "Data Explorer"])

with page[0]:
    st.subheader("Business overview")
    left, right = st.columns(2)
    with left:
        if not filtered_orders.empty:
            monthly = filtered_orders[filtered_orders["status"] == "completed"].groupby("month", as_index=False)["order_amount"].sum()
            fig = px.line(monthly, x="month", y="order_amount", markers=True, title="Completed revenue by month", labels={"month":"Month", "order_amount":"Revenue (INR)"})
            fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (INR)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No order rows match the selected filters.")
    with right:
        if not filtered_orders.empty:
            status_df = filtered_orders["status"].value_counts().rename_axis("status").reset_index(name="orders")
            fig = px.pie(status_df, names="status", values="orders", hole=0.52, title="Order status mix")
            st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        if not customers.empty:
            city_df = customers["city"].value_counts().rename_axis("city").reset_index(name="customers")
            st.plotly_chart(px.bar(city_df, x="city", y="customers", title="Customers by city", labels={"city":"City", "customers":"Customers"}), use_container_width=True)
    with c2:
        if not segment_profile.empty:
            st.plotly_chart(px.bar(segment_profile, x="segment_name", y="size", title="Customer segment sizes", labels={"segment_name":"Segment", "size":"Customers"}, color="segment_name"), use_container_width=True)
    st.subheader("Project health checks")
    checks = [
        ("Part A · Data cleaning", f"{part_a.get('clean_orders', 0):,} cleaned orders and {part_a.get('clean_customers', 0):,} customers"),
        ("Part B · Churn model", f"Random forest test F1: {part_b.get('rf_test', {}).get('f1', 0):.3f}"),
        ("Part C · Ticket classification", f"Few-shot accuracy: {few_shot_accuracy:.1f}% on classified tickets"),
        ("Part D · Policy retrieval", f"With overlap: {part_d.get('policy_evaluation', [{}])[0].get('answerable_fact', 'N/A')} expected facts found"),
    ]
    for title, detail in checks:
        st.markdown(f"**{title}** — {detail}")

with page[1]:
    st.subheader("Order and sales performance")
    if not filtered_orders.empty:
        c1, c2 = st.columns(2)
        with c1:
            cat_orders = filtered_orders.merge(products[["product_id", "category"]], on="product_id", how="left") if not products.empty else filtered_orders.assign(category="Unknown")
            cat_sales = cat_orders[cat_orders["status"] == "completed"].groupby("category", as_index=False)["order_amount"].sum().sort_values("order_amount", ascending=False)
            st.plotly_chart(px.bar(cat_sales, x="category", y="order_amount", title="Revenue by product category", labels={"category":"Category", "order_amount":"Revenue (INR)"}), use_container_width=True)
        with c2:
            if "delivery_minutes" in filtered_orders:
                delivery = filtered_orders[filtered_orders["status"] == "completed"].dropna(subset=["delivery_minutes"])
                st.plotly_chart(px.histogram(delivery, x="delivery_minutes", nbins=25, title="Completed-order delivery time", labels={"delivery_minutes":"Delivery minutes"}), use_container_width=True)
        st.markdown("#### Latest orders in selected view")
        display_cols = [c for c in ["order_id", "customer_id", "product_id", "order_ts", "status", "order_amount", "delivery_minutes"] if c in filtered_orders.columns]
        st.dataframe(filtered_orders[display_cols].sort_values("order_ts", ascending=False).head(100), use_container_width=True, hide_index=True)
        st.download_button("Download filtered orders CSV", filtered_orders.to_csv(index=False).encode("utf-8"), file_name="instamart_filtered_orders.csv", mime="text/csv")
    else:
        st.warning("No order data found. Run `python run_all.py` from the project folder first.")

with page[2]:
    st.subheader("Customer segments and churn risk")
    if not segment_profile.empty:
        st.dataframe(segment_profile.rename(columns={"size":"customers", "mean_recency":"avg_days_since_order", "mean_frequency":"avg_order_frequency", "mean_monetary":"avg_spend_inr"}), use_container_width=True, hide_index=True)
        a, b = st.columns(2)
        with a:
            st.plotly_chart(px.scatter(segment_profile, x="mean_frequency", y="mean_monetary", size="size", color="segment_name", hover_name="segment_name", title="Segment profile: frequency vs spend", labels={"mean_frequency":"Average order frequency", "mean_monetary":"Average spend (INR)", "size":"Customer count"}), use_container_width=True)
        with b:
            st.plotly_chart(px.bar(segment_profile, x="segment_name", y="mean_recency", color="segment_name", title="Average recency by segment", labels={"segment_name":"Segment", "mean_recency":"Days since order"}), use_container_width=True)
    if not segments.empty:
        st.markdown("#### Find a customer")
        customer_search = st.text_input("Search customer ID", placeholder="e.g. C0001")
        seg_view = segments.copy()
        if selected_city != "All cities" and not customers.empty:
            ids = set(customers.loc[customers["city"] == selected_city, "customer_id"])
            seg_view = seg_view[seg_view["customer_id"].isin(ids)]
        if customer_search.strip():
            seg_view = seg_view[seg_view["customer_id"].astype(str).str.contains(customer_search.strip(), case=False)]
        st.dataframe(seg_view.head(200), use_container_width=True, hide_index=True)
        st.download_button("Download customer segments CSV", seg_view.to_csv(index=False).encode("utf-8"), file_name="instamart_customer_segments.csv", mime="text/csv")
    if not retention.empty:
        st.markdown("#### Retention follow-up list")
        st.caption("These are the top 10 saved retention candidates from the model output; validate before using for real customer outreach.")
        st.dataframe(retention, use_container_width=True, hide_index=True)
    chosen = part_b.get("chosen_test", {})
    if chosen:
        st.markdown("#### Churn model quality (test set)")
        metrics = pd.DataFrame([{"Metric":"Precision", "Value":chosen.get("precision", 0)}, {"Metric":"Recall", "Value":chosen.get("recall", 0)}, {"Metric":"F1 score", "Value":chosen.get("f1", 0)}, {"Metric":"Accuracy", "Value":chosen.get("accuracy", 0)}])
        st.plotly_chart(px.bar(metrics, x="Metric", y="Value", range_y=[0,1], title="Selected model test metrics", text=metrics["Value"].map(lambda v:f"{v:.2f}")), use_container_width=True)

with page[3]:
    st.subheader("Support ticket intelligence")
    if not tickets.empty:
        classified = tickets[tickets["category"].fillna("").astype(str).str.len() > 0]
        a, b, c = st.columns(3)
        a.metric("Total tickets", f"{len(tickets):,}")
        b.metric("Classified tickets", f"{len(classified):,}")
        c.metric("Few-shot accuracy", f"{few_shot_accuracy:.1f}%")
        l, r = st.columns(2)
        with l:
            cat_df = tickets["category"].replace("", "Unclassified").value_counts().rename_axis("category").reset_index(name="tickets")
            st.plotly_chart(px.bar(cat_df, x="category", y="tickets", color="category", title="Tickets by category"), use_container_width=True)
        with r:
            route_df = tickets["route"].value_counts().rename_axis("route").reset_index(name="tickets")
            st.plotly_chart(px.pie(route_df, names="route", values="tickets", hole=.45, title="Support routing distribution"), use_container_width=True)
        filter_cat = st.selectbox("Filter ticket category", ["All"] + sorted(tickets["category"].fillna("Unclassified").replace("", "Unclassified").unique().tolist()))
        filter_priority = st.selectbox("Filter priority", ["All"] + sorted(tickets["priority"].dropna().unique().tolist()))
        ticket_view = tickets.copy()
        ticket_view["category_display"] = ticket_view["category"].replace("", "Unclassified").fillna("Unclassified")
        if filter_cat != "All": ticket_view = ticket_view[ticket_view["category_display"] == filter_cat]
        if filter_priority != "All": ticket_view = ticket_view[ticket_view["priority"] == filter_priority]
        st.dataframe(ticket_view, use_container_width=True, hide_index=True)
        st.download_button("Download filtered ticket results CSV", ticket_view.to_csv(index=False).encode("utf-8"), file_name="instamart_ticket_results.csv", mime="text/csv")
        status_counts = part_c.get("few_shot", {}).get("status_counts", {})
        if status_counts:
            st.markdown("#### Classification status")
            sc = pd.DataFrame([{"status":k,"tickets":v} for k,v in status_counts.items()])
            st.dataframe(sc, use_container_width=True, hide_index=True)
    else:
        st.warning("Ticket outputs are missing. Run `python run_all.py` first.")

with page[4]:
    st.subheader("Policy retrieval and support agent")
    pe = part_d.get("policy_evaluation", [])
    if pe:
        st.markdown("#### Retrieval evaluation")
        pe_df = pd.DataFrame([{"chunk overlap":x.get("overlap"), "chunks":x.get("chunks"), "document hit":x.get("answerable_doc_hit"), "fact accuracy":x.get("answerable_fact"), "safe fallback":x.get("unanswerable_fallback")} for x in pe])
        st.dataframe(pe_df, use_container_width=True, hide_index=True)
        st.caption("The saved evaluation found that overlapping policy chunks retained the expected facts better than the no-overlap comparison.")
    scenarios = part_d.get("scenario_results", [])
    if scenarios:
        st.markdown("#### Agent scenario outcomes")
        sdf = pd.DataFrame(scenarios)
        display = [c for c in ["scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "expected_outcome", "expected_refund_inr"] if c in sdf.columns]
        st.dataframe(sdf[display], use_container_width=True, hide_index=True)
        if "status" in sdf:
            outcome = sdf["status"].value_counts().rename_axis("outcome").reset_index(name="scenarios")
            st.plotly_chart(px.bar(outcome, x="outcome", y="scenarios", color="outcome", title="Agent outcomes by scenario"), use_container_width=True)
    st.warning("Agent outcomes and refund amounts shown here are simulated test scenarios. No actual payment or refund has been made.")
    if not part_d:
        st.info("Part D results are not present yet. Run `python run_all.py` first.")

with page[5]:
    st.subheader("Data explorer")
    data_options = {
        "Clean orders": DATA / "orders.csv",
        "Clean customers": DATA / "customers.csv",
        "Clean products": DATA / "products.csv",
        "Customer segments": OUT / "customer_segments.csv",
        "Retention list": OUT / "retention_list.csv",
        "Ticket triage results": OUT / "triage_results.csv",
        "Budget demo results": OUT / "triage_budget_demo.csv",
    }
    selected_data = st.selectbox("Choose a dataset", list(data_options.keys()))
    df = read_csv(data_options[selected_data])
    if not df.empty:
        st.write(f"**{len(df):,} rows × {len(df.columns)} columns**")
        st.dataframe(df.head(1000), use_container_width=True, hide_index=True)
        st.download_button("Download selected dataset CSV", df.to_csv(index=False).encode("utf-8"), file_name=data_options[selected_data].name, mime="text/csv")
    else:
        st.info("This dataset is not available yet. Run the project pipeline first.")

st.divider()
st.caption("Instamart Customer Intelligence & Support Copilot · Local demo dashboard · Synthetic data only")
