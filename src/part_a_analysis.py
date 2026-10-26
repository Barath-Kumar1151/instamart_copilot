from pathlib import Path
import sqlite3,json
import pandas as pd, numpy as np
R=Path(__file__).resolve().parents[1]; DB=R/'data/instamart.db'; O=R/'outputs'
def run():
 sql=(R/'sql/queries.sql').read_text(); statements=[]
 for block in sql.split(';'):
  lines=[ln for ln in block.splitlines() if not ln.strip().startswith('--') and ln.strip()]
  if lines: statements.append('\n'.join(lines))
 results={}
 with sqlite3.connect(DB) as con:
  for i,q in enumerate(statements,1):
   df=pd.read_sql_query(q,con); results[f'Q{i}']={'rows':len(df),'columns':list(df.columns)}
   if i==2: results['Q2']['revenue_sum']=round(float(df.revenue.sum()),2); results['Q2']['largest_row']=df.loc[df.revenue.idxmax()].to_dict()
   if i==5: results['Q5']['average_customer_spend']=round(float(pd.read_sql_query("WITH spend AS (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id) SELECT AVG(total_spend) x FROM spend",con).iloc[0,0]),2)
   if i==6:
    results['Q6']['rank_diff_rows']=int((df.spend_rank!=df.row_num).sum()); results['Q6']['top_by_city']=df.sort_values(['city','spend_rank']).groupby('city').first().reset_index()[['city','customer_id','total_spend']].to_dict('records')
   if i==7: results['Q7']['last_cumulative']=round(float(df.cumulative_revenue.iloc[-1]),2)
  pivot=pd.read_sql_query("SELECT c.city,p.category,SUM(o.order_amount) revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category",con)
  results['pandas_pivot_grand_total']=round(float(pivot.revenue.sum()),2)
  city=pd.read_sql_query("SELECT c.city,SUM(o.order_amount) total,AVG(o.order_amount) mean,COUNT(*) count FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.city",con)
  results['city_stats']=city.to_dict('records')
  orders=pd.read_sql_query("SELECT order_amount,discount_pct,quantity,delivery_minutes FROM orders WHERE status='completed'",con)
 amounts=orders.order_amount; q1,q3=amounts.quantile([.25,.75]); iqr=q3-q1; lo=q1-1.5*iqr; hi=q3+1.5*iqr
 results['amount_stats']={'mean':float(amounts.mean()),'median':float(amounts.median()),'skew':float(amounts.skew()),'q1':float(q1),'q2':float(amounts.median()),'q3':float(q3),'lower_fence':float(lo),'upper_fence':float(hi),'outliers_below':int((amounts<lo).sum()),'outliers_above':int((amounts>hi).sum()),'correlation':orders.corr(numeric_only=True).to_dict()}
 (O/'part_a_analysis.json').write_text(json.dumps(results,indent=2,default=float)); print(json.dumps(results,indent=2,default=float))
if __name__=='__main__': run()
