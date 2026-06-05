"""
Complete Example: New vs Continuing Conversations

This shows how the conversation_id flow works in practice.
"""

import requests

BASE_URL = "http://localhost:8000"
API_KEY = "rsk_your_api_key_here"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


def query_agentic(question: str, conversation_id: str = None):
    """Send an agentic query."""
    payload = {
        "question": question,
        "tool_mode": "strict",
        "return_metadata": False
    }
    
    # Only include conversation_id if provided
    if conversation_id:
        payload["conversation_id"] = conversation_id
    
    response = requests.post(
        f"{BASE_URL}/api/query/agentic",
        headers=headers,
        json=payload
    )
    
    return response.json()


print("=" * 70)
print("SCENARIO 1: New Conversation (No conversation_id provided)")
print("=" * 70)

# First request - NO conversation_id provided
print("\n[Request 1] Client sends question WITHOUT conversation_id")
print("Question: 'What is the refund policy?'")
print("Payload: { 'question': '...', 'conversation_id': null }")

response1 = query_agentic("What is the refund policy?")

print("\n[Response 1] Server generates NEW conversation_id and returns it")
print(f"Answer: {response1['answer'][:100]}...")
print(f"conversation_id: {response1.get('conversation_id')}")  # ← Server generated this
print("^ Client should STORE this conversation_id")

# Store the conversation_id (in real app: sessionStorage, localStorage, etc.)
stored_conversation_id = response1.get('conversation_id')

print("\n" + "=" * 70)
print("SCENARIO 2: Continue Conversation (conversation_id provided)")
print("=" * 70)

# Second request - WITH conversation_id (continuing conversation)
print("\n[Request 2] Client sends follow-up WITH stored conversation_id")
print("Question: 'What about exchanges?'")
print(f"Payload: {{ 'question': '...', 'conversation_id': '{stored_conversation_id}' }}")

response2 = query_agentic("What about exchanges?", stored_conversation_id)

print("\n[Response 2] Server uses SAME conversation_id (continues conversation)")
print(f"Answer: {response2['answer'][:100]}...")
print(f"conversation_id: {response2.get('conversation_id')}")  # ← Same as before
print("^ Agent has context from 'refund policy' question")

print("\n" + "=" * 70)
print("SCENARIO 3: New Conversation (Start Fresh)")
print("=" * 70)

# Third request - NO conversation_id (new conversation)
print("\n[Request 3] Client wants NEW conversation, doesn't send conversation_id")
print("Question: 'How do I use the API?'")
print("Payload: { 'question': '...', 'conversation_id': null }")

response3 = query_agentic("How do I use the API?")

print("\n[Response 3] Server generates DIFFERENT conversation_id")
print(f"Answer: {response3['answer'][:100]}...")
print(f"conversation_id: {response3.get('conversation_id')}")  # ← New ID
print("^ This is a completely separate conversation")

new_conversation_id = response3.get('conversation_id')

print("\n" + "=" * 70)
print("SCENARIO 4: Multiple Active Conversations")
print("=" * 70)

# Continue FIRST conversation
print(f"\n[Request 4a] Continue FIRST conversation (ID: {stored_conversation_id[:8]}...)")
print("Question: 'How long does the refund take?'")
response4a = query_agentic("How long does the refund take?", stored_conversation_id)
print(f"conversation_id: {response4a.get('conversation_id')[:8]}...")
print("^ Uses context from 'refund policy' and 'exchanges'")

# Continue SECOND conversation
print(f"\n[Request 4b] Continue SECOND conversation (ID: {new_conversation_id[:8]}...)")
print("Question: 'What are the rate limits?'")
response4b = query_agentic("What are the rate limits?", new_conversation_id)
print(f"conversation_id: {response4b.get('conversation_id')[:8]}...")
print("^ Uses context from 'API' question, NOT from refund questions")

print("\n" + "=" * 70)
print("CLIENT-SIDE IMPLEMENTATION")
print("=" * 70)

print("""
// JavaScript Example (Web App)
// =============================

let conversationId = sessionStorage.getItem('conversation_id');

async function askQuestion(question) {
    const payload = {
        question: question,
        tool_mode: 'strict'
    };
    
    // Include conversation_id if we have one
    if (conversationId) {
        payload.conversation_id = conversationId;
    }
    
    const response = await fetch('/api/query/agentic', {
        method: 'POST',
        headers: {
            'X-API-Key': apiKey,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });
    
    const data = await response.json();
    
    // IMPORTANT: Store the conversation_id from response
    if (data.conversation_id) {
        conversationId = data.conversation_id;
        sessionStorage.setItem('conversation_id', conversationId);
    }
    
    return data;
}

// Start new conversation
function startNewConversation() {
    conversationId = null;
    sessionStorage.removeItem('conversation_id');
}

// Usage:
// First question (no conversation_id) → Server generates one
await askQuestion("What is the refund policy?");

// Follow-up (uses stored conversation_id) → Server continues conversation
await askQuestion("What about exchanges?");

// Start fresh
startNewConversation();
await askQuestion("How do I use the API?");
""")

print("\n" + "=" * 70)
print("KEY POINTS")
print("=" * 70)
print("""
1. Client does NOT provide conversation_id on FIRST request
   → Server generates new UUID and returns it

2. Client STORES conversation_id from response
   → sessionStorage, localStorage, database, etc.

3. Client PROVIDES conversation_id on FOLLOW-UP requests
   → Server loads context from that conversation

4. Client OMITS conversation_id to start NEW conversation
   → Server generates new UUID (fresh start)

5. Server ALWAYS returns conversation_id in response
   → Client always knows which conversation it's in

6. Multiple conversations can be active simultaneously
   → Each has its own isolated context
""")
