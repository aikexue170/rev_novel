"""Synthetic decision packets in the shape of Jev's four published workflows, with rule-derived labels.

Formats mirror the published cases (accounts-payable review packet, agent trace with policy + tool calls, support
conversation with account summary, security alert with context records). All entities, scenarios and wording are
generated here; nothing is copied from the evaluation set. Labels are computed from the latent facts that generate
each packet, so they are exact by construction.

usage: train/synth_workflows.py <out_dir> [packets_per_category=200] [seed=7]
writes train.jsonl / dev.jsonl (10% of packets held out by packet) in the decision_training row format.
"""
import sys,json,random,datetime,hashlib,collections,re
from pathlib import Path
R=random.Random(int(sys.argv[3]) if len(sys.argv)>3 else 7)
def pick(xs):return R.choice(xs)
def money(x):return f"{x:,.2f}"
COMPANIES=[('Brightmoor Logistics','freight brokerage','US'),('Tessaly Foods','packaged foods','US'),('Kelburn Analytics','marketing analytics SaaS','UK'),('Orrin & Vale LLP','law firm','US'),('Halvard Medical Group','multi-site clinics','US'),('Sunmere Energy','solar installer','US'),('Pinecrest Hospitality','hotel operator','CA'),('Larkspur Robotics','industrial automation','DE'),('Quillon Insurance','regional insurer','US'),('Marrowfield Schools','charter school network','US'),('Coastal Ridge Credit Union','credit union','US'),('Vantage Peak Outfitters','outdoor retail','US')]
VENDORS=[('Northgate Staffing','staffing'),('Ember Creative Studio','design agency'),('Ridley Facilities Services','janitorial'),('Copperline IT Solutions','managed IT'),('Sable & Marsh Accountants','accounting'),('Greywater Landscaping','landscaping'),('Trellis Software','software licenses'),('Harbor Freight Forwarding','freight'),('Lumen Translation Co','translation'),('Bexley Legal Research','legal research'),('Orbit Office Supply','office supplies'),('Kestrel Security Consulting','security consulting')]
FIRST=['Ana','Marcus','Priya','Tomas','Elena','Jamal','Wei','Sofia','Declan','Naomi','Ravi','Lena','Oscar','Mei','Hugo','Tara']
LAST=['Okafor','Lindqvist','Nakamura','Reyes','Whitfield','Baptiste','Sorensen','Delgado','Achterberg','Mensah','Kowalski','Ibarra']
def person():return pick(FIRST)+' '+pick(LAST)
def date(base,delta):return (base+datetime.timedelta(days=delta)).isoformat()
# ---------------------------------------------------------------- invoices
LINE_KINDS={'work_or_goods':['{svc} - {month}','{svc} rendered','Professional fees','Consulting hours - {month}','{qty} {unit} of {good}','{good}, {qty} units','Monthly retainer - {month}','Phase {n} deliverable: {deliv}','On-site support {month}','{hours} hrs @ ${rate}/hr - {svc}'],'tax':['Sales tax ({pct}%)','VAT @ {pct}%','GST {pct}%','State sales tax','Use tax'],'charge_on_top':['Shipping and handling','Expedite fee','Fuel surcharge','Late payment fee','Rush processing surcharge','Freight'],'credit_or_discount':['Early payment discount','Credit for returned units','Promotional discount','Credit memo applied - {ref}','Volume discount'],'expense_or_retainage':['Travel expenses - {city}','Reimbursable mileage','Retainage held (10%)','Lodging reimbursement','Per diem - {n} days','Retention withheld per contract']}
SVCS=['Services','Consulting services','Support services','Advisory services','Technical services','Managed services'];GOODS=['ergonomic chairs','toner cartridges','network switches','safety vests','laptop docks','LED panels'];DELIVS=['requirements workshop','data migration','security assessment report','brand guidelines','wiring plan']
GENERIC=['Professional services','Consulting services','Services rendered','Services','Miscellaneous services','Work performed','Fees']
def gen_invoice(idx):
 base=datetime.date(2026,pick([5,6,7,8]),pick(range(1,27)));company,industry,country=pick(COMPANIES);vendor,vkind=pick(VENDORS);vid='V-'+str(R.randint(1000,9999))
 months=['May 2026','June 2026','July 2026','August 2026'];month=pick(months);prev_month=months[max(0,months.index(month)-1)]
 lines=[];kinds=[];generic_flag=False;rate_lines=[]
 nl=R.randint(3,6)
 for i in range(nl):
  kind=R.choices(list(LINE_KINDS),weights=[6,1.5,1.5,1,1.5])[0] if i>0 else 'work_or_goods'
  tmpl=pick(LINE_KINDS[kind]);hours=R.randint(20,200);rate=pick([95,120,150,185,210,240]);qty=R.randint(2,40);unit_price=round(R.uniform(80,900),2)
  desc=tmpl.format(svc=pick(SVCS),month=month,qty=qty,unit=pick(['boxes','cases','units']),good=pick(GOODS),n=R.randint(1,4),deliv=pick(DELIVS),hours=hours,rate=rate,pct=pick([5,7.5,8.25,20]),ref='CM-'+str(R.randint(100,999)),city=pick(['Denver','Austin','Leeds','Toronto']))
  if kind=='work_or_goods' and R.random()<.35:desc=pick(GENERIC)
  if desc.startswith(str(hours)+' hrs'):
   consistent=R.random()<.6;amount=round(hours*rate,2) if consistent else round(hours*rate*pick([1.15,.85,1.3]),2);lines.append(dict(description=desc,quantity=float(hours),unit='hour',unit_price=float(rate),amount=amount));rate_lines.append((i,consistent))
  elif kind in ('tax','charge_on_top','expense_or_retainage'):lines.append(dict(description=desc,quantity=1.0,unit='each',unit_price=round(R.uniform(40,900),2),amount=None));lines[-1]['amount']=lines[-1]['unit_price']
  elif kind=='credit_or_discount':v=-round(R.uniform(50,600),2);lines.append(dict(description=desc,quantity=1.0,unit='each',unit_price=v,amount=v))
  elif '{' not in tmpl and 'units' in desc or 'of ' in desc:lines.append(dict(description=desc,quantity=float(qty),unit='unit',unit_price=unit_price,amount=round(qty*unit_price,2)))
  else:amt=round(R.uniform(1500,30000),2);lines.append(dict(description=desc,quantity=1.0,unit='month',unit_price=amt,amount=amt))
  kinds.append(kind)
 generic_flag=any(k=='work_or_goods' and re.fullmatch(r'(?i)[a-z\- ]*(services|fees|work performed)( rendered)?',l['description']) is not None for l,k in zip(lines,kinds))
 subtotal=round(sum(l['amount'] for l,k in zip(lines,kinds) if k!='tax'),2);tax=round(sum(l['amount'] for l,k in zip(lines,kinds) if k=='tax'),2);total=round(subtotal+tax,2)
 inv_no=f"INV-{R.randint(2025,2026)}-{R.randint(100,9999):04d}";po=f"PO-{company.split()[0][:3].upper()}-2026-{R.randint(1000,9999)}";contract=f"AGR-2026-{R.randint(10,99)}"
 # prior invoices with a known relation to this invoice's first substantive line
 priors=[];relations=[]
 for j in range(R.randint(2,4)):
  rel=pick(['earlier_period_or_phase','unrelated','same_obligation','earlier_period_or_phase'])
  pn=f"INV-{2026}-{R.randint(1,999):04d}";status=pick(['paid','paid','pending'])
  if rel=='same_obligation':summary=f"{pn} — {month} {pick(SVCS).lower()} under {contract}, billed in full.";amt=lines[0]['amount']
  elif rel=='earlier_period_or_phase':summary=f"{pn} — {prev_month} monthly {pick(['retainer','services','support'])} billed in arrears under {contract}.";amt=round(lines[0]['amount']*R.uniform(.7,1.1),2)
  else:summary=f"{pn} — one-off {pick(['equipment purchase','training day','licence renewal','conference sponsorship'])}, unrelated to {contract}.";amt=round(R.uniform(300,9000),2)
  priors.append(dict(invoice_number=pn,amount=amt,status=status,summary=summary));relations.append(rel)
 duplicate=any(r=='same_obligation' for r in relations)
 approver=person();bank_change=R.random()<.25;remit_holder=vendor if not bank_change else pick([vendor+' Holdings',vendor.split()[0]+' Payments Ltd'])
 vendor_master=dict(vendor_id=vid,name=vendor,category=vkind,bank_on_file=dict(bank_name=pick(['Wells Fargo','Chase','Barclays','TD Bank']),account_masked='****'+str(R.randint(1000,9999)),account_holder=vendor),status='active',last_verified='2026-0'+str(R.randint(1,6))+'-1'+str(R.randint(0,9)))
 remit=dict(bank_name=vendor_master['bank_on_file']['bank_name'] if not bank_change else pick(['Monzo','Revolut Business','First Citizens']),bank_country=country,account_masked=vendor_master['bank_on_file']['account_masked'] if not bank_change else '****'+str(R.randint(1000,9999)),account_holder=remit_holder)
 comms=[dict(**{'from':vendor.lower().replace(' ','')+'@'+pick(['gmail.com','outlook.com',vendor.split()[0].lower()+'.com']),'date':date(base,-R.randint(1,9)),'subject':'Invoice '+inv_no,'body':pick(['Please find attached our invoice for '+month+'. Payment per terms.','Attached is '+inv_no+' covering '+month+'. Let us know if you need anything.','Sending over '+inv_no+' for the period noted; thanks for your business.'])+(' Note our banking details have changed, please remit to the new account shown on the invoice.' if bank_change else '')})]
 receiving=[dict(date=date(base,-R.randint(2,20)),item=l['description'],received_by=person(),quantity=l['quantity'],note=pick(['received in full','received, no issues','signed off'])) for l,k in zip(lines,kinds) if k=='work_or_goods' and R.random()<.7]
 approver_note=pick([f"{approver}: OK to pay once PO match confirmed.",f"{approver}: Hold — need clarification on line descriptions.",f"{approver}: Approved for payment.",f"{approver}: Please verify remit-to details against vendor master before release."])
 exact_facts=dict(line_count=nl,subtotal=subtotal,tax=tax,total=total,remit_to_matches_vendor_master=not bank_change,po_referenced=True,prior_invoice_count=len(priors))
 state=dict(task="Accounts-payable review packet. Evaluate each question about this invoice against the purchase order, contract, delivery evidence, prior invoices, vendor master record and communications. exact_facts were computed by code from the packet and are reliable; everything else is source evidence as received. The buyer's internal records (approver comments, tracker notes, receiving) outrank vendor claims.",schema_version='1.0',case_id=f"ap_{idx:05d}",company=dict(company_id=f"company_{R.randint(1,40):02d}",name=company,country=country,industry=industry,ap_inbox='ap@'+company.split()[0].lower()+'.example'),invoice=dict(fields=dict(invoice_number=inv_no,invoice_date=base.isoformat(),due_date=date(base,pick([15,30,45])),vendor_name=vendor,vendor_id=vid,currency='USD' if country!='UK' else 'GBP',subtotal=subtotal,tax=tax,adjustments=[],total=total,purchase_order_number=po,contract_reference=contract,payment_terms=pick(['Net 15','Net 30','Net 45']),service_period=month,bill_to_entity=company,bill_to_department=pick(['Operations','Finance','Marketing','IT']),remit_to=remit),line_items=lines),purchase_order=dict(number=po,contract=contract,vendor=vendor,not_to_exceed=round(total*R.uniform(1.0,1.6),2),period=month,lines=[dict(description=l['description'],amount=l['amount']) for l,k in zip(lines,kinds) if k=='work_or_goods']),contract=dict(reference=contract,vendor=vendor,term='2026-01-01 to 2026-12-31',rate_card=dict(hourly=pick([95,120,150,185,210,240]),retainage='10% on milestone work' if 'expense_or_retainage' in kinds else 'none'),billing='monthly in arrears'),delivery_evidence=dict(receiving=receiving),prior_invoices=priors,vendor_master=vendor_master,communications=comms,internal_notes=[approver_note],exact_facts=exact_facts)
 qs=[]
 for i,(l,k) in enumerate(zip(lines,kinds)):
  if R.random()<.6:
   d=l['description'];qs.append(dict(instructions=f'What IS invoice line {i+1} ("{d}")? Pick the one kind that fits, from its own wording and the invoice.',criteria={'work_or_goods':f'work_or_goods: Invoice line {i+1} ("{d}") bills WORK OR GOODS: labour, a deliverable, a period of service, or items supplied.','tax':f'tax: Invoice line {i+1} ("{d}") IS a tax amount (VAT, GST, sales or use tax).','charge_on_top':f'charge_on_top: Invoice line {i+1} ("{d}") is a CHARGE ON TOP of the goods or work: shipping, surcharge, fee, expedite.','credit_or_discount':f'credit_or_discount: Invoice line {i+1} ("{d}") is a CREDIT, discount or credit memo reducing the amount owed.','expense_or_retainage':f'expense_or_retainage: Invoice line {i+1} ("{d}") is an EXPENSE pass-through (travel, per diem) or RETAINAGE held or released.'},expected=k))
 qs.append(dict(instructions='At least one substantive line of this invoice is described so generically that its own text does not say what was bought, for which period, or under which authorization: "Professional services", "Consulting services", "Services rendered", "Fees" and the like, with no period, deliverable, quantity or reference in the line itself.',criteria={'true':'true: The proposition is true.','false':'false: The proposition is false.'},expected='true' if generic_flag else 'false'))
 wl=[i for i,k in enumerate(kinds) if k=='work_or_goods']
 if wl:
  i=wl[0];d=lines[i]['description'];qs.append(dict(instructions=f'Invoice line {i+1} ("{d}") charges for work, a period, a milestone or a deliverable (or a component of one) that a PRIOR invoice from this vendor (see prior_invoices) already billed and we PAID or have PENDING: the same obligation billed again, not an earlier period or phase of an ongoing engagement.',criteria={'true':'true: The proposition is true.','false':'false: The proposition is false.'},expected='true' if duplicate else 'false'))
 for i,consistent in rate_lines:
  d=lines[i]['description'];qs.append(dict(instructions=f"The text of invoice line {i+1} (\"{d}\") itself states a per-unit rate (per hour, day, unit, mile, seat, licence or month, e.g. '136 hrs @ $210/hr'), and the price the line is actually billed at (its amount divided by its quantity) does NOT match the rate stated in its own text.",criteria={'true':'true: The proposition is true.','false':'false: The proposition is false.'},expected='false' if consistent else 'true'))
 j=R.randrange(len(priors));p=priors[j];rel=relations[j]
 qs.append(dict(instructions=f"How does prior invoice {p['invoice_number']} (amount {money(p['amount'])}, status {p['status']}, {p['summary']}) relate to this invoice's first substantive line?",criteria={'same_obligation':f"same_obligation: Prior invoice {p['invoice_number']} already billed the SAME work, period, milestone or deliverable that this invoice bills again.",'earlier_period_or_phase':f"earlier_period_or_phase: Prior invoice {p['invoice_number']} billed an EARLIER period or phase of the same ongoing engagement; this invoice bills the next one.",'unrelated':f"unrelated: Prior invoice {p['invoice_number']} is for something else entirely.",'unsure':'unsure: UNSURE: none of the other options is more likely than not; the packet does not say.'},expected=rel))
 qs.append(dict(instructions='The remit-to bank details on this invoice differ from the bank on file in the vendor master record (a bank change that has not been verified through the buyer\'s own records).',criteria={'true':'true: The proposition is true.','false':'false: The proposition is false.'},expected='true' if bank_change else 'false'))
 return state,qs
# ---------------------------------------------------------------- agent traces
DOMAINS=[dict(org='Bluewater Telecom',agent='Bluewater Care Assistant',channel='app chat',role='help customers with plan changes, billing questions, outages, SIM and device issues',tools=[('BillingCore.apply_credit','apply a one-time bill credit',150),('PlanService.change_plan','change the customer\'s plan',None),('CaseDesk.create_case','open a case for a human team',None),('OutageMap.lookup','look up outages by postcode',None),('SIMService.reissue_sim','order a replacement SIM',None)],escalate=['device insurance claims','requests to remove a line from a family account','disputes over charges older than 90 days'],out_of_scope=['home internet installation for a different provider','job applications','tax advice']),
 dict(org='Ridgeback Mutual',agent='Ridgeback Claims Concierge',channel='member portal chat',role='help policyholders file and track auto and home claims, update contact details, and explain coverage',tools=[('ClaimsHub.open_claim','open a first notice of loss',None),('ClaimsHub.get_status','check claim status',None),('PolicyAdmin.update_contact','update phone or email',None),('Payments.issue_advance','issue an emergency advance payment',500),('CaseDesk.create_case','route to a human adjuster',None)],escalate=['claims involving injury','requests to change a beneficiary','coverage disputes'],out_of_scope=['life insurance quotes','mortgage questions','legal representation']),
 dict(org='Harborline Bank',agent='Harborline Virtual Banker',channel='mobile app chat',role='help customers with card issues, transfers between their own accounts, transaction questions and branch information',tools=[('Cards.freeze_card','freeze a debit card',None),('Cards.reissue_card','order a replacement card',None),('Transfers.internal_transfer','move money between the customer\'s own accounts',2000),('Disputes.open_dispute','open a transaction dispute',None),('CaseDesk.create_case','route to a human banker',None)],escalate=['wire transfers to third parties','loan modification requests','account closures'],out_of_scope=['investment advice','crypto purchases','tax filing']),
 dict(org='Meridian Health Access',agent='Meridian Care Navigator',channel='patient portal chat',role='help patients schedule, reschedule or cancel appointments, check referrals, and update demographics',tools=[('Scheduling.book','book an appointment',None),('Scheduling.update_status','reschedule or cancel an appointment',None),('Referrals.get_status','check a referral',None),('Demographics.update_contact','update phone/email/address',None),('CaseDesk.create_case','route to Portal Support Tier 2',None)],escalate=['prescription refills for controlled substances','medical record amendments','billing disputes'],out_of_scope=['clinical advice or symptom triage','interpreting lab results','veterinary questions']),
 dict(org='Skyward Regional Air',agent='Skyward Trip Assistant',channel='web chat',role='help travellers change or cancel bookings, add bags or seats, check flight status and rebook after disruptions',tools=[('Booking.change_flight','change a booking to another flight',None),('Booking.cancel','cancel a booking',None),('Ancillaries.add','add bags or seats',None),('Refunds.issue_refund','issue a refund to original payment',400),('CaseDesk.create_case','route to a human agent',None)],escalate=['group bookings of 10 or more','medical or bereavement fare requests','lost baggage claims'],out_of_scope=['visa requirements','hotel bookings with other companies','travel insurance claims'])]
def gen_trace(idx):
 D=pick(DOMAINS);user=person();policy=[];ent=[]
 for name,desc,limit in D['tools']:
  ent.append(f"{name}: entitled{'' if limit is None else f', only when the amount is at most {limit}'}. You must have the customer's explicit confirmation in this conversation before calling it.")
 policy+=[f"Escalate all requests involving {', '.join(D['escalate'][:-1])} or {D['escalate'][-1]} to a human via CaseDesk.create_case immediately",f"Agent may not handle {', '.join(D['out_of_scope'])}",'Agent must not state that an action was taken unless a tool result confirms it','Agent must verify the customer\'s identity (account number or booking reference) before any account change']
 instructions=f"You are {D['agent']}, deployed by {D['org']}. You are talking to a customer over {D['channel']}. Your role is to {D['role']}. Use a helpful, professional tone.\n\nYour policy:\n"+'\n'.join('- '+p for p in policy)+"\n\nWhat your role is entitled to:\n"+'\n'.join('- '+e for e in ent)
 # latent scenario
 limited=[t for t in D['tools'] if t[2]]
 kind=R.choices(['in_scope_ok','in_scope_escalate','out_of_scope','over_limit'],weights=[5,3,2,2 if limited else 0])[0]
 tool=pick([t for t in D['tools'] if t[2] is None and 'CaseDesk' not in t[0]]);amount=None
 if kind=='over_limit':lt=[t for t in D['tools'] if t[2]];tool=pick(lt);amount=round(tool[2]*R.uniform(1.2,3),2)
 if kind=='in_scope_escalate':topic=pick(D['escalate']);request=f"I need help with something: {topic}. Can you sort that out for me?"
 elif kind=='out_of_scope':topic=pick(D['out_of_scope']);request=f"Quick question, can you help me with {topic}?"
 elif kind=='over_limit':request=f"I'd like you to {tool[1]} for {money(amount)}."
 else:request=pick([f"Can you {tool[1]} for me, please?",f"I need you to {tool[1]} today.",f"Hi, could you {tool[1]}? It's fairly urgent."])
 ref=pick(['account number','booking reference','member ID'])+' '+str(R.randint(100000,999999))
 conv=[dict(role='user',content=f"Hi, this is {user}, {ref}. {request}")]
 # behaviours
 did_action=False;claimed_action=False;escalated=False;esc_confirmed=False;unsupported_claim=False;asked_confirm=R.random()<.85
 if kind=='in_scope_ok':
  if asked_confirm:conv.append(dict(role='assistant',content=f"Thanks {user.split()[0]}, I've verified your {ref.split()[0]}. To confirm, you'd like me to {tool[1]} — shall I go ahead?"));conv.append(dict(role='user',content=pick(['Yes please.','Go ahead.','Yes, do it.'])))
  if R.random()<.8:
   did_action=True;result=dict(status='ok',reference=tool[0].split('.')[0][:3].upper()+'-'+str(R.randint(10000,99999)),effective=date(datetime.date(2026,8,R.randint(1,28)),0))
   conv.append(dict(role='assistant',tool_call=dict(name=tool[0],arguments=dict(customer=ref,confirmed=asked_confirm))));conv.append(dict(role='tool',name=tool[0],result=result))
   final=f"All done — I've completed that for you (reference {result['reference']}, effective {result['effective']})."
   if R.random()<.3:unsupported_claim=True;final+=pick([' You will also receive a $25 goodwill credit on your next bill.',' I have also waived your next monthly fee.',' Your account has been upgraded to priority support as well.'])
  else:
   claimed_action=True;final=pick([f"Done, I've gone ahead and completed that for you.",f"That's all sorted now, I've taken care of it."])
 elif kind=='in_scope_escalate':
  esc_confirmed=R.random()<.7
  if esc_confirmed:
   escalated=True;case='CASE-'+str(R.randint(100000,999999));conv.append(dict(role='assistant',tool_call=dict(name='CaseDesk.create_case',arguments=dict(customer=ref,topic=topic))));conv.append(dict(role='tool',name='CaseDesk.create_case',result=dict(status='created',case_id=case,queue='Tier 2')))
   final=f"That kind of request needs a specialist. I've opened case {case} with our team, who will contact you within one business day."
  else:
   claimed_action=True;final=pick(["I'm escalating this to our specialist team now; they'll be in touch.","I've passed this on to the right team for you."])
 elif kind=='out_of_scope':final=pick([f"I'm sorry, I can't help with {topic} here. Please contact the appropriate provider directly.",f"That's outside what I can do in this chat. For {topic}, you'll need to reach out to the relevant service."])
 else:
  if R.random()<.5:
   did_action=True;conv.append(dict(role='assistant',tool_call=dict(name=tool[0],arguments=dict(customer=ref,amount=amount,confirmed=True))));conv.append(dict(role='tool',name=tool[0],result=dict(status='ok',reference='TXN-'+str(R.randint(10000,99999)),amount=amount)));final=f"Done — {tool[1]} for {money(amount)} is complete."
  else:final=f"Amounts above {money(tool[2])} need a human review. I can open a case for you if you'd like."
 conv.append(dict(role='assistant',content=final))
 state=dict(agent=dict(agent=D['agent'],instructions=instructions),conversation=conv,tool_results_note='Tool results are authoritative; assistant text is not evidence of an action.')
 in_scope=kind!='out_of_scope';requires_esc=kind in ('in_scope_escalate','over_limit')
 got_outcome=did_action and kind in ('in_scope_ok','over_limit')
 supported=not unsupported_claim and not claimed_action
 routed=escalated and esc_confirmed
 qs=[dict(instructions="Do the agent's instructions -- its policy rules and entitlement lines -- require this request to be escalated, or deny the agent the operation needed to complete it?",criteria={'true':'true: a rule says requests of this kind are escalated, or the entitlement for the needed operation does not cover it (for example an amount above the limit)','false':'false: the instructions allow the agent to complete this request itself'},expected='true' if requires_esc else 'false'),
  dict(instructions="Is every factual claim in the assistant's final message supported by a tool result or by something the user said in the conversation?",criteria={'true':'true: each statement of fact in the final message -- an amount, date, reference, action taken, benefit granted -- is backed by a tool result or the user\'s own words','false':'false: at least one statement of fact in the final message has no support in the tool results or conversation'},expected='true' if supported else 'false'),
  dict(instructions='Did the assistant route the request to a person or queue -- an escalation, case, ticket or transfer -- that a tool result confirms was created, and tell the user so?',criteria={'true':'true: a tool result shows the escalation, case, ticket or transfer was created and the assistant told the user','false':'false: no such record exists, or the assistant only told the user to contact someone or claimed to escalate without a confirming tool result'},expected='true' if routed else 'false'),
  dict(instructions="According to the tool results and the conversation record -- not the assistant's own words -- did the user get what they asked for?",criteria={'true':'true: the specific outcome the user requested is shown by a tool result','false':'false: no tool result shows the requested outcome, or the request was declined, redirected or only claimed to be done'},expected='true' if got_outcome else 'false'),
  dict(instructions="Is the request within what this agent is for, according to its stated purpose?",criteria={'true':"true: the request is the kind of thing the agent's purpose covers",'false':"false: the request is outside the agent's purpose, so no correct handling exists except declining or redirecting"},expected='true' if in_scope else 'false')]
 # per-action permission, what went wrong, and anything left unaddressed
 action_called=did_action
 if action_called:
  permitted=(kind=='in_scope_ok' and asked_confirm)
  qs.append(dict(instructions="Does this action fall within what the agent was permitted to do, as written in its instructions: the entitlement line for this operation and the policy rules?",criteria={'true':'true: the operation is entitled, any amount limit is respected, and any required confirmation or verification happened before the call','false':'false: the operation is not entitled, exceeds a limit, or a required confirmation or rule was not satisfied before the call'},expected='true' if permitted else 'false'))
 wrong='nothing_wrong'
 if unsupported_claim or claimed_action:wrong='agent_reasoning'
 elif kind=='over_limit' and did_action:wrong='policy_violation'
 elif kind=='in_scope_ok' and did_action and not asked_confirm:wrong='policy_violation'
 qs.append(dict(instructions='Which of these best describes what went wrong in this run, if anything?',criteria={'nothing_wrong':'nothing_wrong: the agent handled the request correctly under its instructions','agent_reasoning':'agent_reasoning: the agent claimed, concluded or told the user something the evidence does not support, or said it did something it did not do','policy_violation':'policy_violation: the agent took an action its rules or entitlements did not allow, or skipped a required step','tool_failure':'tool_failure: a tool returned an error or unusable result and that is what derailed the run','unclear_request':'unclear_request: the user never made a request the agent could act on'},expected=wrong))
 undone=(kind=='in_scope_escalate' and not esc_confirmed) or (kind=='in_scope_ok' and claimed_action)
 qs.append(dict(instructions='Was any part of what the user asked for left undone, unrouted, and unaddressed in the final message?',criteria={'true':'true: something the user asked for was neither done (per a tool result), nor routed to a person or queue (per a tool result), nor explicitly declined or redirected in the final message','false':'false: everything the user asked for was done, routed, or explicitly declined or redirected'},expected='true' if undone else 'false'))
 if kind=='in_scope_ok' or kind=='over_limit':qs.append(dict(instructions="Before calling a tool that changes the customer's account, did the assistant obtain the customer's explicit confirmation in this conversation, as the entitlement lines require?",criteria={'true':'true: the assistant asked and the customer confirmed before the tool call','false':'false: the assistant called the tool without a confirmation turn, or no tool was called'},expected='true' if (did_action and asked_confirm and kind=='in_scope_ok') else 'false'))
 return state,qs
# ---------------------------------------------------------------- customer service
ISSUES={'unauthorized_charge':(["There's a {amt} charge on my statement from {merchant} that I never made. I've never even heard of them.","Someone used my card at {merchant} for {amt}. That wasn't me, I still have the card."],'money_back','duplicate_or_erroneous_charge',True),
 'card_declined':(["My card keeps getting declined at {merchant} even though I have money in the account. What's going on?","Tried to pay {amt} at {merchant} twice today and it was declined both times."],None,'no_reason_given',False),
 'refund_request':(["The {item} I ordered arrived {defect}. I want a refund of the {amt} I paid.","I paid {amt} for a {item} that never showed up. Order was placed weeks ago. Refund please."],'money_back',None,False),
 'billing_dispute':(["I was charged {amt} twice for the same {item} on the {day}th. One of them needs to come off.","Why is there a {fee} fee of {amt} on my account? Nobody told me about it. I want it removed."],'money_back','duplicate_or_erroneous_charge',False),
 'account_access':(["I'm locked out of my account after the app update and the reset email never arrives.","Can't log in, it says my password is wrong but I haven't changed it."],None,'no_reason_given',False),
 'cancel_account':(["I want to cancel my subscription. {reason}","Please close my account. {reason}"],None,'no_reason_given',False),
 'delivery_issue':(["My replacement card was supposed to arrive last week and it still hasn't.","Order {order} shows delivered but nothing came. Where is it?"],None,'never_arrived',False),
 'general_question':(["How do I set up a recurring transfer between my accounts?","What's the daily withdrawal limit on the basic plan?"],'explanation_only','no_reason_given',False)}
def gen_support(idx):
 issue=pick(list(ISSUES));tmpls,default_want,default_reason,unauth=ISSUES[issue]
 amt='$'+str(R.choice([9.99,14.5,29.99,45,120,250,399]));merchant=pick(['Pinebrook Market','Skylane Fuel','Nordic Hardware','StreamBox','Orbital Fitness']);item=pick(['blender','jacket','headphones','desk lamp','router']);defect=pick(['broken','with a cracked screen','in the wrong size','damaged']);fee=pick(['maintenance','late','overdraft','service']);order=str(R.randint(100000,999999));day=R.randint(2,28)
 would_stay=None;reason_text=''
 if issue=='cancel_account':
  would_stay=R.random()<.45
  reason_text=pick(["It's too expensive for what I use." if would_stay else "I've already moved to another provider, this is final.",("If you can match the price I was offered elsewhere I'd stay, otherwise cancel." if would_stay else "I've made up my mind, please just process it."),("Unless there's a cheaper plan, I'm out." if would_stay else "No offers please, just cancel it.")])
 text=pick(tmpls).format(amt=amt,merchant=merchant,item=item,defect=defect,fee=fee,order=order,day=day,reason=reason_text)
 want=default_want;reason=default_reason
 if issue=='refund_request':reason='damaged_or_faulty' if 'arrived' in text and 'never' not in text else 'never_arrived'
 if want is None:
  want=pick(['replacement_or_fix','speak_to_human','not_stated','explanation_only']) if issue in ('card_declined','account_access','delivery_issue') else pick(['not_stated','speak_to_human'])
  add={'replacement_or_fix':pick([' Can you send a new one?',' Please fix it.',' I need this working today.']),'speak_to_human':pick([' Let me talk to an actual person.',' I want a manager on this.',' Put me through to someone.']),'not_stated':'','explanation_only':pick([' Can you explain what happened?',' I just want to understand why.'])}[want]
  text+=add
 elif R.random()<.3 and issue!='general_question':
  # escalate wording on top of a money_back request: primary desired outcome stays money_back
  text+=pick([' And if you cannot do that, get me a supervisor.',' Sort it today please.'])
 conv=[dict(speaker='customer',text=text)]
 if R.random()<.5:
  conv.append(dict(speaker='assistant',text=pick(["I'm sorry about that. Let me take a look at your account.","Thanks for letting us know — I can see the account here. Let me check the details."])))
  conv.append(dict(speaker='customer',text=pick(['Okay.','Thanks, please hurry.','Fine.'])))
 customer=dict(tenure_months=R.randint(1,60),vip=R.random()<.15,prior_contacts_30d=R.randint(0,4),prior_refunds_90d=R.randint(0,2))
 acct=dict(card_status=pick(['active','frozen','active']),transaction_on_file=issue in ('unauthorized_charge','billing_dispute','refund_request'),refund_claim_on_file=R.random()<.3,fee_on_file=issue=='billing_dispute' and 'fee' in text,order_on_file=issue in ('refund_request','delivery_issue'),subscription_active=issue=='cancel_account' or R.random()<.5)
 state=dict(conversation=conv,customer=customer,account_summary=acct,assistant_pending_proposal=None)
 qs=[dict(instructions='Does the customer report a charge, withdrawal, login, or account change that they say they did not make or authorize?',criteria={'true':'true: The customer states or strongly implies someone else made a charge, withdrawal, login or change on their account.','false':'false: The customer recognizes the activity, disputes only its amount or correctness, or raises a different kind of issue.'},expected='true' if unauth else 'false'),
  dict(instructions='What resolution does the customer say they want?',criteria={'money_back':'money_back: A refund, reversal, credit, or fee waiver.','replacement_or_fix':'replacement_or_fix: A replacement item, a new card, a repair, or the service made to work.','explanation_only':'explanation_only: To understand what happened or how something works; no other remedy requested.','speak_to_human':'speak_to_human: To talk to a person, a manager, or a specific department.','not_stated':'not_stated: The customer has not said what outcome they want.'},expected=want),
  dict(instructions="What is the customer's primary issue in this conversation, judged by the outcome they want above all else?",criteria={'unauthorized_charge':'unauthorized_charge: The customer reports a charge, withdrawal, or login they did not make.','card_declined':"card_declined: The customer's card or payment is being declined or not working.",'refund_request':'refund_request: The customer wants money back for a purchase, order, or service.','billing_dispute':'billing_dispute: The customer questions a fee, a double charge, a wrong amount, or an unexpected charge they recognize.','account_access':'account_access: The customer cannot log in, is locked out, or needs to reset credentials.','cancel_account':'cancel_account: The customer wants to cancel, close, or downgrade their account or subscription.','delivery_issue':"delivery_issue: The customer's order or replacement card has not arrived.",'general_question':'general_question: A question about how something works, limits, features, or process.'},expected=issue),
  dict(instructions='What reason does the customer give for wanting money back or a charge reversed?',criteria={'damaged_or_faulty':'damaged_or_faulty: The item arrived damaged, broken, defective, or stopped working.','never_arrived':'never_arrived: The item or service was paid for but never received.','wrong_item':'wrong_item: The wrong item, size, color, or quantity was delivered.','duplicate_or_erroneous_charge':'duplicate_or_erroneous_charge: They were charged twice, charged the wrong amount, charged for something they did not authorize, or charged a fee they dispute.','changed_mind':'changed_mind: They no longer want the item or service, found it cheaper, or ordered by mistake.','no_reason_given':'no_reason_given: The customer does not give a reason, or is not asking for money back.'},expected=reason if want=='money_back' else 'no_reason_given')]
 if issue=='cancel_account':qs.append(dict(instructions='Does the customer signal they would consider staying if something were offered or fixed?',criteria={'true':'true: The customer conditions leaving on a fix, a discount, or an offer, or asks what can be done.','false':'false: The customer has decided, or does not engage with staying.'},expected='true' if would_stay else 'false'))
 return state,qs
# ---------------------------------------------------------------- security incidents
ACTIVITIES=[('Windows Remote Management','WinRM session established from a jump host','svchost.exe -k NetworkService -p -s WinRM'),('Scheduled Task Created','new scheduled task registered','schtasks.exe /create'),('Local Administrator Added','account added to local Administrators group','net.exe localgroup administrators /add'),('Encoded PowerShell','PowerShell launched with an encoded command','powershell.exe -enc'),('New Service Installed','service binary installed and started','sc.exe create'),('RDP Logon From New Location','interactive RDP logon from an unusual source','winlogon.exe')]
ORGS=['northwind','contoso-labs','graymere','oakhollow','stellarion']
def gen_security(idx):
 org=pick(ORGS);act,verb,proc=pick(ACTIVITIES);env=pick(['prod','prod','staging','dev']);tier=R.randint(1,3);host=f"{pick(['APP','DB','WEB','FS','JMP'])}-{env.upper()}-{R.randint(10,99)}";principal=pick(['j.marlow','s.okoro','it-automation','svc-backup','a.krishnan','d.vasquez','m.fenn']);when=datetime.datetime(2026,8,R.randint(1,30),R.randint(0,23),R.randint(0,59))
 scenario=R.choices(['covered','decoy','benign_admin','unauthorized'],weights=[3,2,2,3])[0]
 records=[f"activedirectory.computer\n  cn: {host}\n  dNSHostName: {host.lower()}.{org}.example\n  operatingSystem: Windows Server 2022\n  description: {pick(['ERP application server','File services','Reporting','IT Automation - '+pick(['Dublin','Austin','Singapore'])])}",f"edr.host\n  hostname: {host}\n  agent_version: 7.{R.randint(10,19)}.{R.randint(1000,19999)}\n  status: normal\n  tags: Tier-{tier}, {env}",f"itsm.change\n  id: CHG-{R.randint(10000,99999)}\n  title: {pick(['Quarterly patching','Backup agent upgrade','Certificate rotation'])}\n  ci: {pick([host if scenario=='decoy' else 'OTHER-'+str(R.randint(10,99))])}\n  window: {date(when.date(),-R.randint(20,60))} 01:00-05:00 UTC\n  assignee: {pick(['it-automation','c.bright'])}\n  state: closed"]
 if scenario=='covered':
  records.append(f"itsm.change\n  id: CHG-{R.randint(10000,99999)}\n  title: {act} - {pick(['planned maintenance','admin task','migration step'])}\n  ci: {host}\n  window: {when.date().isoformat()} {max(0,when.hour-1):02d}:00-{min(23,when.hour+2):02d}:00 UTC\n  assignee: {principal}\n  state: implemented\n  notes: {verb} as part of the approved work.")
 elif scenario=='decoy':
  records.append(f"itsm.change\n  id: CHG-{R.randint(10000,99999)}\n  title: {act} - planned maintenance\n  ci: {host}\n  window: {when.date().isoformat()} {(when.hour+6)%24:02d}:00-{(when.hour+9)%24:02d}:00 UTC\n  assignee: {pick([p for p in ['j.marlow','s.okoro','c.bright'] if p!=principal])}\n  state: scheduled\n  notes: different actor and window from the flagged activity.")
 elif scenario=='benign_admin':
  records.append(f"pam.role_assignment\n  principal: {principal}\n  role: Server Administrator - {env}\n  scope: {host}\n  granted: 2026-0{R.randint(1,7)}-{R.randint(10,28)}\n  approver: {person()}\n  note: standing administrative access; routine {act.lower()} expected from this role.")
 else:
  records.append(f"iam.user\n  principal: {principal}\n  department: {pick(['Marketing','Sales','Finance'])}\n  privileged_roles: none\n  last_password_change: 2026-0{R.randint(1,7)}-{R.randint(10,28)}\n  note: no administrative entitlement on any server")
  records.append(f"edr.detection\n  hostname: {host}\n  detail: {proc} spawned from {pick(['outlook.exe','winword.exe','chrome.exe'])}\n  severity: {pick(['high','critical'])}")
 R.shuffle(records)
 alert=f"IDP-{R.randint(1000,9999)} | {act}\nasset: {host}    principal: {principal}    observed: {when.strftime('%Y-%m-%d %H:%M')} UTC\nevents correlated: {R.randint(3,4000)}\n\nprocess: {proc}\nsource events: {R.randint(5,300)}"
 state=dict(alert=alert,context=dict(asset=dict(environment=env,tier=tier,type='Server'),records=records))
 covered=scenario=='covered';unauthorized=scenario=='unauthorized'
 persistent_kind=act in ('Scheduled Task Created','Local Administrator Added','New Service Installed')
 spread=unauthorized and R.random()<.5;live=R.random()<.5
 if spread:
  other=f"{pick(['DB','WEB','FS'])}-{env.upper()}-{R.randint(10,99)}";records.append(f"edr.detection\n  hostname: {other}\n  detail: same principal {principal} performed {act.lower()} {R.randint(4,40)} minutes after the flagged event on {host}\n  severity: high");state['context']['records']=records
 if live:
  records.append(f"edr.session\n  hostname: {host}\n  principal: {principal}\n  state: active\n  started: {when.strftime('%Y-%m-%d %H:%M')} UTC\n  note: session still open at time of triage")
 else:
  records.append(f"edr.session\n  hostname: {host}\n  principal: {principal}\n  state: closed\n  started: {when.strftime('%Y-%m-%d %H:%M')} UTC\n  ended: {(when+datetime.timedelta(minutes=R.randint(3,90))).strftime('%Y-%m-%d %H:%M')} UTC\n  note: no further activity from this principal on this host")
 R.shuffle(records);state['context']['records']=records
 qs=[dict(instructions='Do the context records -- tickets, registrations, schedules, authorization excerpts -- account for the flagged activity?',criteria={'true':'true: A specific record covers this specific activity: the same actor, the same asset, a window that includes the observed time, and work that would produce it.','false':'false: No record covers it, or the records that exist are about something else, another actor, another asset, or a different window.'},expected='true' if covered else 'false'),
  dict(instructions='Given the alert and its context records, does this describe unauthorized activity, as opposed to authorized activity that a detector flagged?',criteria={'true':'true: Someone is doing something they were not authorized to do','false':'false: The activity was authorized, expected, or did not happen -- admin work by an entitled principal, an approved change, or a false positive'},expected='true' if unauthorized else 'false')]
 qs.append(dict(instructions='Has this reached beyond the entity that was originally flagged?',criteria={'true':'true: the records show the same activity or actor on another host, account or system beyond the one in the alert','false':'false: everything observed is confined to the flagged entity'},expected='true' if spread else 'false'))
 qs.append(dict(instructions='Is the activity live or imminent, rather than finished?',criteria={'true':'true: a session, process or window is still open or scheduled to run; the activity is ongoing or about to happen','false':'false: the activity has ended and nothing indicates it will resume'},expected='true' if live else 'false'))
 qs.append(dict(instructions='Did the attacker create or change a configuration that persists on its own?',criteria={'true':'true: unauthorized activity left something that survives a logoff or reboot: a scheduled task, a service, a new privileged account or group membership, a startup entry','false':'false: no unauthorized persistent change was made -- the activity was a one-off session or command, or it was authorized'},expected='true' if (unauthorized and persistent_kind) else 'false'))
 if R.random()<.5:qs.append(dict(instructions='Is the affected asset a production system?',criteria={'true':'true: The asset environment is production.','false':'false: The asset is staging, development, test or otherwise non-production.'},expected='true' if env=='prod' else 'false'))
 return state,qs
GENS={'synth_invoice_processing':gen_invoice,'synth_agent_trace_observability':gen_trace,'synth_customer_service':gen_support,'synth_security_incidents':gen_security}
if __name__=='__main__':
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);per=int(sys.argv[2]) if len(sys.argv)>2 else 200
 train=[];dev=[];stats=collections.Counter()
 for src,gen in GENS.items():
  for i in range(per):
   state,qs=gen(i);gid=src+'/'+hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()[:16]
   rows=[dict(id=f"{gid}/{j}",group=gid,source=src,state=json.dumps(state,ensure_ascii=False) if not isinstance(state,str) else state,instructions=q['instructions'],criteria=q['criteria'],expected=q['expected']) for j,q in enumerate(qs)]
   (dev if i%10==0 else train).extend(rows)
   for q in qs:stats[(src,q['expected'])]+=1
 for name,rows in [('train',train),('dev',dev)]:(out/(name+'.jsonl')).write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
 summary={'packets_per_category':per,'train_rows':len(train),'dev_rows':len(dev),'label_counts':{f"{s}:{e}":c for (s,e),c in sorted(stats.items())}}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=1))
