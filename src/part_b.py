from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
R=Path(__file__).resolve().parents[1]; O=R/'outputs'; O.mkdir(exist_ok=True)
def run():
 c=pd.read_csv(R/'data/clean/customers.csv'); o=pd.read_csv(R/'data/clean/orders.csv',parse_dates=['order_ts']); p=pd.read_csv(R/'data/clean/products.csv')
 o=o.merge(p[['product_id','category']],on='product_id',how='left')
 comp=o[o.status.eq('completed')].copy(); snap=pd.Timestamp('2025-10-01')
 rfm=comp.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index(); rfm['recency']=(snap-rfm.last_order).dt.days; rfm=rfm.drop(columns='last_order')
 scaler=StandardScaler(); X=scaler.fit_transform(rfm[['recency','frequency','monetary']])
 km_rows=[]
 for k in range(2,9):
  km=KMeans(n_clusters=k,n_init=10,random_state=42); labels=km.fit_predict(X); km_rows.append({'k':k,'inertia':km.inertia_,'silhouette':silhouette_score(X,labels)})
 pd.DataFrame(km_rows).to_csv(O/'kmeans_metrics.csv',index=False)
 k=4; model=KMeans(n_clusters=k,n_init=10,random_state=42); rfm['segment_id']=model.fit_predict(X)
 prof=rfm.groupby('segment_id').agg(size=('customer_id','size'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean')).reset_index()
 premium_id=int(prof.loc[prof.mean_monetary.idxmax(),'segment_id'])
 # Names are business descriptions, independent of numeric cluster label ordering.
 names={}
 for _,z in prof.iterrows():
  if int(z.segment_id)==premium_id: name='Premium High-Value'
  elif z.mean_frequency>=prof.mean_frequency.quantile(.75): name='Frequent Essentials'
  elif z.mean_recency>=prof.mean_recency.quantile(.75): name='At-Risk / Win-Back'
  else: name='Occasional Shoppers'
  names[int(z.segment_id)]=name
 rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_id.eq(premium_id)
 rfm[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']].to_csv(O/'customer_segments.csv',index=False)
 prof['segment_name']=prof.segment_id.map(names); prof['is_premium']=prof.segment_id.eq(premium_id); prof.to_csv(O/'segment_profile.csv',index=False)
 pca=PCA(n_components=2).fit(X); (O/'pca.json').write_text(json.dumps({'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'total':float(pca.explained_variance_ratio_.sum())},indent=2))
 # Lapse cohort/features: strict pre-cutoff order-only features.
 cutoff=pd.Timestamp('2025-09-01'); pre=o[(o.order_ts<cutoff)].copy(); pre_comp=pre[pre.status.eq('completed')]; cohort_ids=set(pre_comp.customer_id)
 september=set(comp.loc[(comp.order_ts>=cutoff)&(comp.order_ts<pd.Timestamp('2025-10-01')),'customer_id'])
 feat=pre_comp.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean'))
 feat['recency']=(cutoff-feat.last_order).dt.days; feat=feat.drop(columns='last_order')
 final_count=pre[pre.status.isin(['completed','cancelled'])].groupby('customer_id').status.count(); cancel_count=pre[pre.status.eq('cancelled')].groupby('customer_id').size(); feat['cancellation_rate']=(cancel_count.reindex(feat.index,fill_value=0)/final_count.reindex(feat.index,fill_value=0)).fillna(0)
 fav=pre_comp.groupby('customer_id')['category'].agg(lambda x:x.value_counts().index[0]); feat['favorite_category']=fav
 feat=feat.join(c.set_index('customer_id')[['city','age','gender','signup_date']]); feat['tenure_days']=(cutoff-pd.to_datetime(feat.signup_date)).dt.days; feat=feat.drop(columns='signup_date')
 feat['lapsed']=[int(cid not in september) for cid in feat.index]
 # fill every feature explicitly, including possible category gaps
 num=[x for x in feat.columns if x not in ['favorite_category','city','gender','lapsed']]
 for col in num: feat[col]=feat[col].fillna(feat[col].median())
 for col in ['favorite_category','city','gender']: feat[col]=feat[col].fillna('unknown')
 Xdf=feat.drop(columns='lapsed'); y=feat.lapsed
 assert not Xdf.isna().any().any(), 'Feature matrix contains missing values'
 X_train,X_test,y_train,y_test=train_test_split(Xdf,y,test_size=.2,stratify=y,random_state=42)
 numeric=[x for x in Xdf.columns if x not in ['favorite_category','city','gender']]; categorical=['city','gender','favorite_category']
 preprocessor=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),numeric),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(drop='first',handle_unknown='ignore'))]),categorical)])
 baseline=np.zeros(len(y_test),dtype=int)
 lr=Pipeline([('prep',preprocessor),('model',LogisticRegression(max_iter=1000))]); gs_lr=GridSearchCV(lr,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1); gs_lr.fit(X_train,y_train)
 rf=Pipeline([('prep',preprocessor),('model',RandomForestClassifier(n_estimators=100,random_state=42))]); gs_rf=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1); gs_rf.fit(X_train,y_train)
 def metrics(est):
  yp=est.predict(X_test); return {'confusion_matrix':confusion_matrix(y_test,yp).tolist(),'precision':precision_score(y_test,yp,zero_division=0),'recall':recall_score(y_test,yp,zero_division=0),'f1':f1_score(y_test,yp,zero_division=0),'accuracy':accuracy_score(y_test,yp)}
 chosen_is_rf = metrics(gs_rf.best_estimator_)['f1'] >= metrics(gs_lr.best_estimator_)['f1']
 chosen=gs_rf.best_estimator_ if chosen_is_rf else gs_lr.best_estimator_
 # If needed, logistic model chosen from grid object or estimator both support predict.
 joblib.dump(chosen,O/'lapse_model.joblib'); loaded=joblib.load(O/'lapse_model.joblib'); reproduced=bool(np.array_equal(chosen.predict(X_test),loaded.predict(X_test)))
 # Oversample minority class on train only for logistic comparison.
 from sklearn.utils import resample
 train=pd.concat([X_train,y_train.rename('lapsed')],axis=1); counts=train.lapsed.value_counts(); maj=int(counts.idxmax()); minc=int(counts.idxmin()); minority=train[train.lapsed==minc]; extra=resample(minority,replace=True,n_samples=int(counts[maj]-counts[minc]),random_state=42); balanced=pd.concat([train,extra]).sample(frac=1,random_state=42)
 from sklearn.base import clone
 bal_lr=clone(gs_lr.best_estimator_); bal_lr.fit(balanced.drop(columns='lapsed'),balanced.lapsed)
 # test-set premium retention list is deliberately post-hoc and illustrative.
 seg=pd.read_csv(O/'customer_segments.csv'); premium=set(seg.loc[seg.is_premium.astype(str).str.lower().isin(['true','1']),'customer_id'])
 prob=chosen.predict_proba(X_test)[:,list(chosen.classes_).index(1)]
 ret=pd.DataFrame({'customer_id':X_test.index,'lapse_probability':prob,'actually_lapsed':y_test.values}); ret=ret[ret.customer_id.isin(premium)].sort_values('lapse_probability',ascending=False).head(10); ret.to_csv(O/'retention_list.csv',index=False)
 # Important features and metrics summary
 if hasattr(chosen,'best_estimator_'): chosen_est=chosen.best_estimator_; chosen_params=chosen.best_params_; chosen_cv=chosen.best_score_
 else: chosen_est=chosen; chosen_params={}; chosen_cv=None
 lr_names=gs_lr.best_estimator_.named_steps['prep'].get_feature_names_out(); lr_coef=gs_lr.best_estimator_.named_steps['model'].coef_[0]; coef_map={str(n):float(v) for n,v in zip(lr_names,lr_coef)}
 report={'rf_best_params':gs_rf.best_params_,'rf_cv_f1':gs_rf.best_score_,'rf_test':metrics(gs_rf.best_estimator_),'lr_best_params':gs_lr.best_params_,'lr_cv_f1':gs_lr.best_score_,'lr_test':metrics(gs_lr.best_estimator_),'balanced_lr_test':metrics(bal_lr),'logistic_coefficients':{k:v for k,v in coef_map.items() if any(x in k for x in ['num__recency','num__frequency','num__cancellation_rate'])},'baseline':{'accuracy':accuracy_score(y_test,baseline),'f1':f1_score(y_test,baseline,zero_division=0)},'cohort_rows':len(feat),'lapsed_count':int(y.sum()),'lapsed_rate':float(y.mean()),'train_rows':len(X_train),'test_rows':len(X_test),'chosen_model':'random_forest' if chosen_is_rf else 'logistic_regression','chosen_test':metrics(chosen),'reload_predictions_identical':reproduced,'premium_retention_rows':len(ret),'premium_retention_actually_lapsed':int(ret.actually_lapsed.sum()),'train_frequency_scaler_mean':float(chosen_est.named_steps['prep'].named_transformers_['num'].named_steps['scaler'].mean_[numeric.index('frequency')]) if 'frequency' in numeric else None,'full_cohort_frequency_mean':float(feat.frequency.mean())}
 # Feature importance, transformed names
 rf_est=gs_rf.best_estimator_; feat_names=rf_est.named_steps['prep'].get_feature_names_out(); imp=rf_est.named_steps['model'].feature_importances_; report['rf_top_features']=[{'feature':str(feat_names[i]),'importance':float(imp[i])} for i in np.argsort(imp)[::-1][:5]]
 (O/'part_b_summary.json').write_text(json.dumps(report,indent=2,default=float))
 print('Part B complete:',json.dumps(report,indent=2,default=float))
if __name__=='__main__': run()
