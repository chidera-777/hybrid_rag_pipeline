# Conversation Flow: How conversation_id Works

## The Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT SIDE                              │
└─────────────────────────────────────────────────────────────────┘

Request 1: NEW CONVERSATION
───────────────────────────
Client: "What is the refund policy?"
conversation_id: null (not provided)
                    │
                    ▼
        ┌───────────────────────┐
        │   RAG API Service     │
        │                       │
        │ 1. No conversation_id │
        │    provided           │
        │ 2. Generate new UUID  │
        │    conv_abc123        │
        │ 3. Create memory      │
        │ 4. Process query      │
        │ 5. Save to DynamoDB   │
        └───────────────────────┘
                    │
                    ▼
Response 1:
{
  "answer": "Refunds available within 30 days...",
  "conversation_id": "conv_abc123"  ← Server generated
}
                    │
                    ▼
Client stores: sessionStorage.setItem('conversation_id', 'conv_abc123')


Request 2: CONTINUE CONVERSATION
─────────────────────────────────
Client: "What about exchanges?"
conversation_id: "conv_abc123" (from storage)
                    │
                    ▼
        ┌───────────────────────┐
        │   RAG API Service     │
        │                       │
        │ 1. conversation_id    │
        │    provided           │
        │ 2. Load from DynamoDB │
        │    - Q: refund policy │
        │    - A: 30 days...    │
        │ 3. Use context        │
        │ 4. Process query      │
        │ 5. Update DynamoDB    │
        └───────────────────────┘
                    │
                    ▼
Response 2:
{
  "answer": "Exchanges allowed within 60 days...",
  "conversation_id": "conv_abc123"  ← Same ID
}
                    │
                    ▼
Client: Already has this ID stored


Request 3: NEW CONVERSATION
───────────────────────────
Client: "How do I use the API?"
conversation_id: null (client wants fresh start)
                    │
                    ▼
        ┌───────────────────────┐
        │   RAG API Service     │
        │                       │
        │ 1. No conversation_id │
        │    provided           │
        │ 2. Generate new UUID  │
        │    conv_xyz789        │
        │ 3. Create NEW memory  │
        │ 4. Process query      │
        │ 5. Save to DynamoDB   │
        └───────────────────────┘
                    │
                    ▼
Response 3:
{
  "answer": "API authentication requires...",
  "conversation_id": "conv_xyz789"  ← Different ID
}
                    │
                    ▼
Client stores: sessionStorage.setItem('conversation_id', 'conv_xyz789')
```

## DynamoDB Storage

```
Table: RAG-ConversationMemory

After Request 1:
┌──────────────┬────────────────────────────────────┬──────────────────┐
│  tenant_id   │            sort_key                │    question      │
├──────────────┼────────────────────────────────────┼──────────────────┤
│ tenant_123   │ conversation#conv_abc123#17371234  │ refund policy?   │
└──────────────┴────────────────────────────────────┴──────────────────┘

After Request 2:
┌──────────────┬────────────────────────────────────┬──────────────────┐
│  tenant_id   │            sort_key                │    question      │
├──────────────┼────────────────────────────────────┼──────────────────┤
│ tenant_123   │ conversation#conv_abc123#17371234  │ refund policy?   │
│ tenant_123   │ conversation#conv_abc123#17371240  │ exchanges?       │
└──────────────┴────────────────────────────────────┴──────────────────┘

After Request 3:
┌──────────────┬────────────────────────────────────┬──────────────────┐
│  tenant_id   │            sort_key                │    question      │
├──────────────┼────────────────────────────────────┼──────────────────┤
│ tenant_123   │ conversation#conv_abc123#17371234  │ refund policy?   │
│ tenant_123   │ conversation#conv_abc123#17371240  │ exchanges?       │
│ tenant_123   │ conversation#conv_xyz789#17371250  │ use API?         │
└──────────────┴────────────────────────────────────┴──────────────────┘
                                    ▲
                                    │
                        Different conversation_id
```

## Decision Logic

```python
# Server-side logic (in pipeline.py)

if conversation_id is None:
    # NEW CONVERSATION
    conversation_id = str(uuid.uuid4())
    print(f"Starting new conversation: {conversation_id}")
    # Creates empty memory
else:
    # CONTINUE CONVERSATION
    print(f"Continuing conversation: {conversation_id}")
    # Loads existing memory from DynamoDB

# Always return conversation_id in response
response["conversation_id"] = conversation_id
```

## Client-Side Patterns

### Pattern 1: Single Active Conversation (Most Common)

```javascript
// Store ONE conversation_id
let conversationId = sessionStorage.getItem('conversation_id');

function ask(question) {
    const payload = { question };
    if (conversationId) {
        payload.conversation_id = conversationId;
    }
    
    const response = await fetch('/api/query/agentic', {
        method: 'POST',
        body: JSON.stringify(payload)
    });
    
    const data = await response.json();
    
    // Update stored ID
    conversationId = data.conversation_id;
    sessionStorage.setItem('conversation_id', conversationId);
    
    return data;
}

function resetConversation() {
    conversationId = null;
    sessionStorage.removeItem('conversation_id');
}
```

### Pattern 2: Multiple Active Conversations

```javascript
// Store MULTIPLE conversation_ids
const conversations = {
    'support': null,
    'sales': null,
    'technical': null
};

function ask(question, channel) {
    const payload = { question };
    
    // Use channel-specific conversation_id
    if (conversations[channel]) {
        payload.conversation_id = conversations[channel];
    }
    
    const response = await fetch('/api/query/agentic', {
        method: 'POST',
        body: JSON.stringify(payload)
    });
    
    const data = await response.json();
    
    // Store per channel
    conversations[channel] = data.conversation_id;
    
    return data;
}

// Usage:
ask("What is pricing?", "sales");      // Conversation A
ask("API issue", "technical");         // Conversation B
ask("Upgrade plan", "sales");          // Continues Conversation A
```

### Pattern 3: Database-Backed (Multi-User)

```python
# Store in database per user
class UserSession(Model):
    user_id = ForeignKey(User)
    conversation_id = UUIDField(null=True)
    last_active = DateTimeField()

def ask_question(user_id, question):
    session = UserSession.objects.get(user_id=user_id)
    
    payload = {"question": question}
    
    # Include conversation_id if exists
    if session.conversation_id:
        payload["conversation_id"] = str(session.conversation_id)
    
    response = requests.post(
        'http://api/query/agentic',
        json=payload
    )
    
    data = response.json()
    
    # Update database
    session.conversation_id = data.get('conversation_id')
    session.last_active = now()
    session.save()
    
    return data

def reset_conversation(user_id):
    session = UserSession.objects.get(user_id=user_id)
    session.conversation_id = None
    session.save()
```

## Summary

| Scenario | Client Sends | Server Does | Server Returns |
|----------|-------------|-------------|----------------|
| **New conversation** | `conversation_id: null` | Generates UUID | New UUID |
| **Continue conversation** | `conversation_id: "abc123"` | Loads context | Same UUID |
| **Start fresh** | `conversation_id: null` | Generates UUID | New UUID |

**Key Insight**: The client controls when to start new conversations by simply NOT sending the conversation_id. The server always returns it so the client can continue if desired.
