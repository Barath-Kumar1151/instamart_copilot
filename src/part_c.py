from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import re,json,random,time
import pandas as pd
import mock_llm
R=Path(__file__).resolve().parents[1]; O=R/'outputs'; RAW=R/'data'/'raw'; MAX_INPUT_TOKENS=400; BASE_DELAY=.05
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'); PHONE=re.compile(r'\b[6-9]\d{9}\b'); ORDER=re.compile(r'ORD-\d{6}')
CATS={'billing','delivery','technical','fraud'}
ZERO='CLASSIFY. Classify the ticket into exactly one allowed category: billing, delivery, technical, fraud. Reply with JSON only in the form {"category": "...", "reason": "..."}.'
FEW='CLASSIFY. Classify the ticket into exactly one allowed category: billing, delivery, technical, fraud. Reply with JSON only in the form {"category": "...", "reason": "..."}.\nEXAMPLES:\nTicket: Unauthorized transaction on my card. Reply: {"category":"fraud","reason":"unauthorized transaction"}\nTicket: My delivery is late. Reply: {"category":"delivery","reason":"late delivery"}'
def redact(t):
 counts={};
 for name,pat,rep in [('email',EMAIL,'[EMAIL]'),('phone',PHONE,'[PHONE]'),('order_id',ORDER,'[ORDER_ID]')]: t,n=pat.subn(rep,t); counts[name]=n
 return t,counts
def classify(text,system,ticket_id):
 messages=[{'role':'system','content':system},{'role':'user','content':text}]
 for retry in range(4):
  try: return mock_llm.chat(messages)
  except mock_llm.TransientLLMError:
   if retry>=3: raise
   wait=BASE_DELAY*(2**retry)*random.uniform(.9,1.15); print(f'ticket_id={ticket_id} retry={retry} wait={wait:.4f}s'); time.sleep(wait)
def run_pipeline(system,budget=None):
 tickets=pd.read_csv(RAW/'tickets.csv',keep_default_na=False); seg=pd.read_csv(O/'customer_segments.csv'); seg['is_premium']=seg.is_premium.astype(str).str.lower().isin(['true','1']); segmap=seg.set_index('customer_id')[['segment_name','is_premium']].to_dict('index')
 mock_llm.reset(); random.seed(42); total=0; rows=[]; redcounts={'email':0,'phone':0,'order_id':0}
 for _,t in tickets.iterrows():
  txt=t['text']; status=''; category=''; used=0; segment=segmap.get(t.customer_id,{}).get('segment_name','')
  if not txt.strip(): status='skipped_empty'
  elif mock_llm.est_tokens(txt)>MAX_INPUT_TOKENS: status='flagged_long'
  else:
   safe,counts=redact(txt)
   for k,v in counts.items(): redcounts[k]+=v
   assert not EMAIL.search(safe) and not PHONE.search(safe) and not ORDER.search(safe)
   if budget is not None and total>=budget: status='budget_exhausted'
   else:
    resp=classify(safe,system,t.ticket_id); used=resp.usage.total_tokens; total+=used
    raw=resp.choices[0].message.content
    try:
     obj=json.loads(raw)
     if isinstance(obj,dict) and obj.get('category') in CATS: category=obj['category']; status='classified_json'
     else: status='flagged_invalid_output'
    except (ValueError,TypeError):
     words=re.findall(r'[a-z]+',(raw or '').lower()); found=[c for c in CATS if c in words]
     if len(found)==1: category=found[0]; status='classified_keyword'
     else: status='flagged_invalid_output'
  if status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}: route='human_review'; priority='NORMAL'
  elif category=='fraud': route='security_team'; priority='HIGH'
  elif segmap.get(t.customer_id,{}).get('is_premium',False): route='premium_desk'; priority='HIGH'
  else: route='support_queue'; priority='NORMAL'
  rows.append({'ticket_id':t.ticket_id,'customer_id':t.customer_id,'status':status,'category':category,'segment_name':segment,'priority':priority,'route':route,'tokens_used':used,'true_category':t.true_category})
 return pd.DataFrame(rows),total,redcounts

def evaluate(df,total):
 classified=df[df.status.isin(['classified_json','classified_keyword'])]
 return {'tickets_total':len(df),'classified':len(classified),'accuracy':float((classified.category==classified.true_category).mean()),'tokens_total':int(total),'tokens_per_classified':round(total/len(classified),2) if len(classified) else 0,'status_counts':df.status.value_counts().to_dict(),'confusion':pd.crosstab(classified.true_category,classified.category).to_dict()}
def run():
 O.mkdir(exist_ok=True)
 z,zt,zc=run_pipeline(ZERO); f,ft,fc=run_pipeline(FEW)
 budget,btotal,_=run_pipeline(ZERO,budget=6000)
 f.drop(columns='true_category').to_csv(O/'triage_results.csv',index=False); budget.drop(columns='true_category').to_csv(O/'triage_budget_demo.csv',index=False)
 report={'zero_shot':evaluate(z,zt),'few_shot':evaluate(f,ft),'budget_demo':{'budget':6000,'exhausted':int((budget.status=='budget_exhausted').sum()),'tokens':btotal},'redactions':fc,'redactions_zero_shot':zc,'remaining_pii_matches':0,'few_shot_status_counts':f.status.value_counts().to_dict()}
 (O/'part_c_summary.json').write_text(json.dumps(report,indent=2))
 print('Part C complete:',json.dumps(report,indent=2))
if __name__=='__main__': run()
