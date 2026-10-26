from pathlib import Path
import pandas as pd, json, sqlite3
R=Path(__file__).resolve().parents[1]
def main():
 c=pd.read_csv(R/'data/clean/customers.csv'); o=pd.read_csv(R/'data/clean/orders.csv'); p=pd.read_csv(R/'data/clean/products.csv')
 checks={
 'customers_1200':len(c)==1200,
 'six_cities':c.city.nunique()==6,
 'no_missing_age':c.age.notna().all(),
 'ages_18_60':c.age.between(18,60).all(),
 'orders_12309':len(o)==12309,
 'unique_order_ids':o.order_id.nunique()==len(o),
 'no_pending':not o.status.eq('pending').any(),
 'statuses_11110_1199':o.status.value_counts().to_dict()=={'completed':11110,'cancelled':1199},
 'amounts_complete':o.order_amount.notna().all(),
 'completed_delivery_complete':o.loc[o.status.eq('completed'),'delivery_minutes'].notna().all(),
 'cancelled_delivery_missing':o.loc[o.status.eq('cancelled'),'delivery_minutes'].isna().all(),
 'completed_revenue':round(o.loc[o.status.eq('completed'),'order_amount'].sum(),2)==2998668.80,
 }
 for k,v in checks.items(): print(f'{k}: {v}')
 if not all(checks.values()): raise AssertionError('One or more checkpoints failed')
 print('ALL CHECKPOINTS PASSED')
if __name__=='__main__': main()
