import os
import logging
import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kanbox-Perplexity Integration", version="1.0.0")

KANBOX_API_KEY = os.environ.get("KANBOX_API_KEY", "")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
KANBOX_API_BASE = "https://api.kanbox.io"
PERPLEXITY_API_BASE = "https://api.perplexity.ai"

def kanbox_headers():
    return {"X-API-Key": KANBOX_API_KEY, "Content-Type": "application/json"}

def perplexity_headers():
    return {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}

async def perplexity_chat(messages: list, model: str = "sonar") -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{PERPLEXITY_API_BASE}/chat/completions",
            headers=perplexity_headers(),
            json={"model": model, "messages": messages}
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

async def kanbox_get(path: str, params: dict = None):
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{KANBOX_API_BASE}{path}", headers=kanbox_headers(), params=params or {})
        resp.raise_for_status()
        return resp.json()

async def kanbox_patch(path: str, body: dict):
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.patch(f"{KANBOX_API_BASE}{path}", headers=kanbox_headers(), json=body)
        resp.raise_for_status()
        return resp.json()

async def enrich_lead(lead: dict) -> dict:
    name = f"{lead.get('firstname', '')} {lead.get('lastname', '')}".strip()
    prompt = f"""Research this LinkedIn professional and provide enrichment data as JSON:
Name: {name}
Headline: {lead.get('headline', '')}
Company: {lead.get('company', '')}
LinkedIn: {lead.get('linkedin_url', lead.get('profileUrl', ''))}

Return ONLY valid JSON with keys: company_summary, role_analysis, recent_news, pain_points, talking_point"""

    content = await perplexity_chat([
        {"role": "system", "content": "You are a B2B sales intelligence assistant. Always respond with valid JSON only."},
        {"role": "user", "content": prompt}
    ], model="sonar")
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        return json.loads(content[start:end]) if start >= 0 else {"raw": content}
    except Exception:
        return {"raw": content}

async def gen_message(lead: dict, enrichment: dict, msg_type: str = "connection_request") -> str:
    types = {
        "connection_request": "LinkedIn connection request. MAX 300 chars. Personal, specific, no pitch.",
        "first_message": "First message after connecting. MAX 500 chars. Reference their work, end with ONE question.",
        "follow_up": "Follow-up message. MAX 400 chars. Different angle, friendly."
    }
    prompt = f"""Write a personalized LinkedIn message:
Name: {lead.get('firstname', 'there')}, Company: {lead.get('company', '')}, Role: {lead.get('headline', '')}
Context: {enrichment.get('company_summary', '')}
Pain points: {enrichment.get('pain_points', '')}
Talking point: {enrichment.get('talking_point', '')}
Type: {types.get(msg_type, types['first_message'])}
Return ONLY the message text."""

    msg = await perplexity_chat([
        {"role": "system", "content": "Expert LinkedIn copywriter. Write authentic non-salesy messages. Never start with Hi [Name] or I hope this finds you well."},
        {"role": "user", "content": prompt}
    ])
    return msg.strip().strip('"').strip("'")

async def process_member(member_id: str):
    try:
        member = await kanbox_get(f"/public/members/{member_id}")
        enrichment = await enrich_lead(member)
        msgs = {t: await gen_message(member, enrichment, t) for t in ["connection_request", "first_message", "follow_up"]}
        import datetime
        notes = f"""=== PERPLEXITY AI ENRICHMENT ===
Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

COMPANY: {enrichment.get('company_summary', 'N/A')}
ROLE: {enrichment.get('role_analysis', 'N/A')}
RECENT NEWS: {enrichment.get('recent_news', 'N/A')}
PAIN POINTS: {enrichment.get('pain_points', 'N/A')}
TALKING POINT: {enrichment.get('talking_point', 'N/A')}

=== AI-GENERATED MESSAGES ===

[CONNECTION REQUEST]
{msgs['connection_request']}

[FIRST MESSAGE]
{msgs['first_message']}

[FOLLOW-UP]
{msgs['follow_up']}"""
        await kanbox_patch(f"/public/members/{member_id}", {"notes": notes})
        logger.info(f"Enriched member {member_id}")
        return {"status": "success", "member_id": member_id, "enrichment": enrichment, "messages": msgs}
    except Exception as e:
        logger.error(f"Error on {member_id}: {e}")
        return {"status": "error", "member_id": member_id, "error": str(e)}

@app.get("/")
async def root():
    return {"status": "running", "service": "Kanbox-Perplexity Integration", "version": "1.0.0",
            "endpoints": ["POST /webhook/kanbox", "POST /enrich/{id}", "POST /enrich/{id}/sync",
                          "GET /enrich/batch", "POST /campaigns/{id}/enrich-leads",
                          "POST /message/generate", "GET /leads", "GET /members",
                          "GET /lists", "GET /campaigns", "GET /stats"]}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/webhook/kanbox")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    logger.info(f"Webhook: {str(body)[:200]}")
    member_id = None
    if isinstance(body, dict):
        member_id = body.get('id') or body.get('member_id') or body.get('linkedin_id') or body.get('memberId')
    elif isinstance(body, list) and body:
        first = body[0]
        member_id = first.get('id') or first.get('member_id') or first.get('linkedin_id')
    if not member_id:
        return JSONResponse({"status": "received", "warning": "no member_id", "payload": body})
    background_tasks.add_task(process_member, str(member_id))
    return JSONResponse({"status": "accepted", "member_id": member_id})

@app.post("/enrich/{member_id}")
async def enrich_async(member_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_member, member_id)
    return {"status": "accepted", "member_id": member_id}

@app.post("/enrich/{member_id}/sync")
async def enrich_sync(member_id: str):
    return await process_member(member_id)

@app.get("/enrich/batch")
async def enrich_batch(background_tasks: BackgroundTasks, list_name: str = None, limit: int = 10):
    params = {"limit": limit}
    if list_name: params["name"] = list_name
    leads = await kanbox_get("/public/leads", params)
    items = leads.get('items', leads.get('leads', leads)) if isinstance(leads, dict) else leads
    queued = [str(l.get('id') or l.get('member_id')) for l in (items or [])[:limit] if l.get('id') or l.get('member_id')]
    for mid in queued:
        background_tasks.add_task(process_member, mid)
    return {"status": "accepted", "queued": len(queued), "member_ids": queued}

@app.get("/leads")
async def get_leads(name: str = None, q: str = None, limit: int = 50):
    params = {"limit": limit}
    if name: params["name"] = name
    if q: params["q"] = q
    return await kanbox_get("/public/leads", params)

@app.get("/members")
async def get_members(limit: int = 50):
    return await kanbox_get("/public/members", {"limit": limit})

@app.get("/lists")
async def get_lists():
    return await kanbox_get("/public/lists")

@app.get("/campaigns")
async def get_campaigns():
    return await kanbox_get("/public/campaigns")

@app.get("/campaigns/{campaign_id}/leads")
async def get_campaign_leads(campaign_id: int, limit: int = 50):
    return await kanbox_get(f"/public/campaigns/{campaign_id}/leads", {"limit": limit})

@app.post("/campaigns/{campaign_id}/enrich-leads")
async def enrich_campaign(campaign_id: int, background_tasks: BackgroundTasks, limit: int = 20):
    data = await kanbox_get(f"/public/campaigns/{campaign_id}/leads", {"limit": limit})
    items = data.get('items', data.get('leads', data)) if isinstance(data, dict) else data
    queued = [str(l.get('id') or l.get('member_id') or l.get('linkedin_id')) for l in (items or []) if l.get('id') or l.get('member_id') or l.get('linkedin_id')]
    for mid in queued:
        background_tasks.add_task(process_member, mid)
    return {"status": "accepted", "campaign_id": campaign_id, "queued": len(queued)}

@app.post("/message/generate")
async def generate_message(request: Request):
    body = await request.json()
    lead = body.get('lead', {})
    if not lead: raise HTTPException(400, "lead required")
    enrichment = body.get('enrichment') or await enrich_lead(lead)
    msg = await gen_message(lead, enrichment, body.get('message_type', 'connection_request'))
    return {"message": msg, "enrichment": enrichment}

@app.get("/stats")
async def get_stats():
    results = {}
    for s in ["requested", "accepted", "messaged", "answered"]:
        try: results[s] = await kanbox_get(f"/public/stats/campaigns/{s}")
        except Exception as e: results[s] = {"error": str(e)}
    return results

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
