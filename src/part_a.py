from pathlib import Path
import sqlite3, json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'/'raw'; CLEAN=ROOT/'data'/'clean'; OUT=ROOT/'outputs'; SQL=ROOT/'sql'
CLEAN.mkdir(parents=True,exist_ok=True); OUT.mkdir(exist_ok=True); SQL.mkdir(exist_ok=True)

def run():
    customers=pd.read_csv(RAW/'customers.csv',keep_default_na=False)
    products=pd.read_csv(RAW/'products.csv')
    orders=pd.read_csv(RAW/'orders.csv',keep_default_na=False)
    # Audit snapshots are saved as text reports for reproducibility.
    audit={}
    for name,df in [('customers',customers),('products',products),('orders',orders)]:
        audit[name]={'shape':list(df.shape),'dtypes':{k:str(v) for k,v in df.dtypes.items()},'missing':{c:{'count':int(df[c].isna().sum()+((df[c]=='').sum() if df[c].dtype=='object' else 0)),'pct':round(100*(df[c].isna().sum()+((df[c]=='').sum() if df[c].dtype=='object' else 0))/len(df),2)} for c in df.columns}}
    audit['specific']={'raw_orders_rows':len(orders),'exact_duplicate_rows':int(orders.duplicated().sum()),'pending_rows':int((orders.status=='pending').sum()),'distinct_order_id':int(orders.order_id.nunique()),'non_numeric_amounts':int(pd.to_numeric(orders.order_amount,errors='coerce').isna().sum()),'negative_delivery':int((pd.to_numeric(orders.delivery_minutes,errors='coerce')<0).sum()),'blank_delivery':int((orders.delivery_minutes=='').sum()),'blank_age':int((customers.age=='').sum()),'age_outside_10_100':int(((pd.to_numeric(customers.age,errors='coerce')<10)|(pd.to_numeric(customers.age,errors='coerce')>100)).sum()),'raw_city_spellings':int(customers.city.nunique()),'raw_gender_spellings_nonblank':int(customers.loc[customers.gender!='','gender'].nunique()),'blank_gender':int((customers.gender=='').sum())}
    (OUT/'raw_audit.json').write_text(json.dumps(audit,indent=2))
    # Customers: standardise city and gender, invalid ages become missing, city median fills.
    customers['city']=customers.city.astype(str).str.strip().str.title()
    customers['gender']=customers.gender.astype(str).str.strip().str.lower().map({'m':'male','male':'male','f':'female','female':'female'}).fillna('unknown')
    customers['age']=pd.to_numeric(customers.age,errors='coerce')
    customers.loc[~customers.age.between(10,100),'age']=float('nan')
    customers['age']=customers['age'].fillna(customers.groupby('city').age.transform('median')).round().astype(int)
    customers['signup_date']=pd.to_datetime(customers.signup_date).dt.strftime('%Y-%m-%d')
    # Orders: remove exact duplicates, then retain latest status record per order.
    orders['order_ts']=pd.to_datetime(orders.order_ts)
    orders=orders.drop_duplicates().sort_values(['order_ts']).drop_duplicates('order_id',keep='last').copy()
    orders['quantity']=pd.to_numeric(orders.quantity).astype(int)
    orders['discount_pct']=pd.to_numeric(orders.discount_pct).astype(float)
    orders['order_amount']=pd.to_numeric(orders.order_amount,errors='coerce')
    orders=orders.merge(products[['product_id','price_inr','category']],on='product_id',how='left',validate='many_to_one')
    missing=orders.order_amount.isna()
    orders.loc[missing,'order_amount']=(orders.loc[missing,'price_inr']*orders.loc[missing,'quantity']*(1-orders.loc[missing,'discount_pct']/100)).round(2)
    orders['delivery_minutes']=pd.to_numeric(orders.delivery_minutes,errors='coerce')
    orders.loc[orders.delivery_minutes<0,'delivery_minutes']=float('nan')
    completed=orders.status.eq('completed')
    med=orders.loc[completed].groupby('category').delivery_minutes.transform('median')
    orders.loc[completed,'delivery_minutes']=orders.loc[completed].groupby('category').delivery_minutes.transform(lambda s:s.fillna(s.median()))
    # robust category median map fills any remaining missing completed rows
    category_medians=orders.loc[completed].groupby('category').delivery_minutes.median()
    orders.loc[completed,'delivery_minutes']=orders.loc[completed,'delivery_minutes'].fillna(orders.loc[completed,'category'].map(category_medians))
    orders.loc[orders.status.eq('cancelled'),'delivery_minutes']=float('nan')
    orders=orders.drop(columns=['price_inr','category'])
    orders['order_ts']=orders.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
    products['price_inr']=pd.to_numeric(products.price_inr)
    customers.to_csv(CLEAN/'customers.csv',index=False); products.to_csv(CLEAN/'products.csv',index=False); orders.to_csv(CLEAN/'orders.csv',index=False)
    db=ROOT/'data'/'instamart.db'; db.parent.mkdir(exist_ok=True)
    with sqlite3.connect(db) as con:
        customers.to_sql('customers',con,if_exists='replace',index=False)
        products.to_sql('products',con,if_exists='replace',index=False)
        orders.to_sql('orders',con,if_exists='replace',index=False)
    # Save audit/checkpoint summary
    summary={'clean_customers':len(customers),'city_counts':customers.city.value_counts().to_dict(),'gender_counts':customers.gender.value_counts().to_dict(),'missing_ages':int(customers.age.isna().sum()),'clean_orders':len(orders),'status_counts':orders.status.value_counts().to_dict(),'completed_amount_sum':round(orders.loc[orders.status=='completed','order_amount'].sum(),2),'missing_completed_delivery':int(orders.loc[orders.status=='completed','delivery_minutes'].isna().sum()),'missing_cancelled_delivery':int(orders.loc[orders.status=='cancelled','delivery_minutes'].isna().sum())}
    (OUT/'part_a_summary.json').write_text(json.dumps(summary,indent=2))
    print('Part A cleaning complete:',summary)
if __name__=='__main__': run()
