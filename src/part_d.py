from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, time, uuid, sqlite3, re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
R=Path(__file__).resolve().parents[1]; RAW=R/'data'/'raw'; POL=R/'data'/'policies'; O=R/'outputs'; STATE=R/'state'; DB=R/'data'/'instamart.db'
O.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
def make_chunks(overlap=10):
 chunks=[]
 for path in sorted(POL.glob('*.txt')):
  words=path.read_text().split(); start=0
  while start<len(words):
   end=min(start+40,len(words)); chunks.append({'source':path.name,'text':' '.join(words[start:end])})
   if end>=len(words): break
   start=end-overlap
 return chunks
def make_retriever(overlap=10):
 chunks=make_chunks(overlap); vec=TfidfVectorizer(); matrix=vec.fit_transform([x['text'] for x in chunks])
 def retrieve(query,k=3):
  q=vec.transform([query]); scores=cosine_similarity(q,matrix)[0]; order=scores.argsort()[::-1][:k]
  return [{'source':chunks[i]['source'],'score':float(scores[i]),'text':chunks[i]['text']} for i in order if scores[i]>0]
 return chunks,vec,matrix,retrieve
def answer(question,retrieve):
 hits=retrieve(question,3); context='\n'.join(f"[{x['source']}] {x['text']}" for x in hits)
 messages=[{'role':'system','content':'POLICY ASSISTANT. Answer only from the supplied context. If the answer is not present, say: I could not find that information in the policy documents.'},{'role':'user','content':f'CONTEXT:\n{context}\nQUESTION:\n{question}'}]
 resp=mock_llm.chat(messages); return resp.choices[0].message.content,[x['source'] for x in hits]
def policy_eval():
 evaldf=pd.read_csv(RAW/'policy_eval.csv',keep_default_na=False); results=[]
 for overlap in (10,0):
  chunks,vec,mat,retrieve=make_retriever(overlap); okdoc=okfact=fallback=0; broken=[]
  for _,row in evaldf.iterrows():
   hits=retrieve(row.question,3); ans,sources=answer(row.question,retrieve)
   if row.expected_doc!='NONE':
    okdoc+=int(row.expected_doc in sources); okfact+=int(row.expected_fact in ans)
    if row.expected_fact not in ans: broken.append(row.question)
   else: fallback+=int(ans.startswith('I could not find'))
  results.append({'overlap':overlap,'chunks':len(chunks),'answerable_doc_hit':f'{okdoc}/12','answerable_fact':f'{okfact}/12','unanswerable_fallback':f'{fallback}/2','failed_questions':broken})
 return results
# Tool functions
TOOLS_SPEC=[
 {'name':'get_order','description':'Look up one order in the SQLite database.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id']}},
 {'name':'search_policy','description':'Retrieve policy passages relevant to a query.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},
 {'name':'calculate_refund','description':'Calculate the refund allowed by the late-delivery policy.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':'number'}},'required':['order_amount','delivery_minutes']}},
 {'name':'issue_refund','description':'Record an automatically approved refund.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},
 {'name':'escalate_to_human','description':'Record a refund that requires a human lead.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},
]
_,_,_,retrieve=make_retriever(10)
def get_order(order_id):
 with sqlite3.connect(DB) as con: row=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
 if not row: return {'error':'order not found'}
 return {'order_id':row[0],'status':row[1],'order_amount':float(row[2]),'delivery_minutes':None if row[3] is None else float(row[3])}
def search_policy(query): return {'results':retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes): return {'refund_inr':round(float(order_amount),2) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0}
def ledger(tool,order_id,refund_inr):
 path=O/'refund_ledger.jsonl'; entry={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'tool':tool,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
 with path.open('a') as f: f.write(json.dumps(entry)+'\n')
 return {'refund_inr':round(float(refund_inr),2)}
def issue_refund(order_id,refund_inr): return ledger('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return ledger('escalate_to_human',order_id,refund_inr)
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}
def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tool=None,resume=False):
 run_id=run_id or f'{scenario_id}_{uuid.uuid4().hex[:8]}'; cp=STATE/f'{run_id}.json'
 if resume:
  saved=json.loads(cp.read_text()); messages=saved['messages']; tokens_used=saved['tokens_used']; steps=saved['steps']; tools_executed=saved['tools_executed']; start=time.monotonic()-saved.get('elapsed_seconds',0)
 else:
  messages=[{'role':'system','content':'AGENT. Follow refund policy and use tools to verify every request.'},{'role':'user','content':user_message}]; tokens_used=0; steps=[]; tools_executed=[]; start=time.monotonic()
 llm_calls=0
 while llm_calls<max_iterations:
  if time.monotonic()-start>=time_limit_seconds:
   return {'status':'FALLBACK_TIMEOUT','final_answer':'FALLBACK_TIMEOUT: hand off to a human agent; time limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'run_id':run_id}
  resp=mock_llm.chat(messages); llm_calls+=1; tokens_used+=resp.usage.total_tokens
  if tokens_used>token_limit:
   return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':'FALLBACK_TOKEN_LIMIT: hand off to a human agent; token limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'run_id':run_id}
  msg=resp.choices[0].message; calls=msg.tool_calls
  if not calls:
   final=msg.content or 'FALLBACK: hand off to a human agent.'; status=final.split(':',1)[0]
   return {'status':status,'final_answer':final,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'run_id':run_id}
  call=calls[0]; args=call.arguments if isinstance(call.arguments,dict) else json.loads(call.arguments)
  messages.append({'role':'assistant','content':None,'tool_calls':[{'id':call.id,'name':call.name,'arguments':args}]})
  observation=TOOL_MAP[call.name](**args); step={'tool':call.name,'arguments':args,'observation':observation}; steps.append(step); tools_executed.append(call.name)
  messages.append({'role':'tool','tool_call_id':call.id,'name':call.name,'content':json.dumps(observation)})
  cp.write_text(json.dumps({'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens_used,'step_number':len(steps),'steps':steps,'tools_executed':tools_executed,'elapsed_seconds':time.monotonic()-start},indent=2))
  if crash_after_tool is not None and len(tools_executed)==crash_after_tool: raise RuntimeError('Simulated crash after checkpoint')
 return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':'FALLBACK_MAX_ITERATIONS: hand off to a human agent; iteration limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'run_id':run_id}
def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
 saved=json.loads((STATE/f'{run_id}.json').read_text())
 # run_agent reloads checkpoint and continues without replaying completed tools.
 return run_agent('',saved['scenario_id'],max_iterations,token_limit,time_limit_seconds,run_id=run_id,resume=True)
def run():
 O.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
 for old_checkpoint in STATE.glob('*.json'): old_checkpoint.unlink()
 (O/'refund_ledger.jsonl').write_text('')
 pe=policy_eval(); (O/'policy_eval_summary.json').write_text(json.dumps(pe,indent=2))
 scenarios=pd.read_csv(RAW/'agent_scenarios.csv',keep_default_na=False); runs=[]
 for _,s in scenarios.iterrows():
  mock_llm.reset(); result=run_agent(s.message,s.scenario_id); runs.append({**result,'scenario_id':s.scenario_id,'expected_outcome':s.expected_outcome,'expected_refund_inr':float(s.expected_refund_inr)})
 # Demonstrations of guardrails
 mock_llm.reset(); token_demo=run_agent(scenarios.iloc[0].message,'S1_TOKEN_DEMO',token_limit=200)
 mock_llm.reset(); timeout_demo=run_agent(scenarios.iloc[0].message,'S1_TIMEOUT_DEMO',time_limit_seconds=0)
 # Crash/resume demonstration. This is a separate simulated run; required scenario ledger entries remain intact.
 s1=scenarios.iloc[0]; mock_llm.reset(); run_id='S1_CRASH_RESUME'
 try: run_agent(s1.message,'S1',run_id=run_id,crash_after_tool=2)
 except RuntimeError: pass
 resumed=resume_agent(run_id)
 # The first two tool executions happened before crash; continuation adds only remaining calls.
 # Run outputs for seven required scenarios are written independently of the crash demo.
 with (O/'agent_runs.jsonl').open('w') as f:
  for r in runs: f.write(json.dumps({k:r[k] for k in ['scenario_id','status','final_answer','llm_calls','tokens_used','steps']},default=str)+'\n')
 summary={'policy_evaluation':pe,'scenario_results':[{'scenario_id':r.get('scenario_id',scenarios.iloc[i].scenario_id),'status':r['status'],'final_answer':r['final_answer'],'llm_calls':r['llm_calls'],'tokens_used':r['tokens_used'],'expected_outcome':r['expected_outcome'],'expected_refund_inr':r['expected_refund_inr']} for i,r in enumerate(runs)],'token_demo_status':token_demo['status'],'timeout_demo_status':timeout_demo['status'],'crash_resume':{'status':resumed['status'],'final_answer':resumed['final_answer'],'tools_executed':resumed['steps'],'run_id':run_id}}
 (O/'part_d_summary.json').write_text(json.dumps(summary,indent=2,default=str))
 print('Part D complete:',json.dumps(summary,indent=2,default=str))
if __name__=='__main__': run()
