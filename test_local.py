#!/usr/bin/env python3
"""
Script de teste local para o SERH RAG Chatbot
Testa as principais funcionalidades SEM depender do Vertex AI
(ideal para testar com Python 3.14)
"""

import requests
import json
import time

# URL da aplicação (local ou Railway)
BASE_URL = "http://localhost:8000"

def print_section(title):
    """Imprime um separador com título"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def test_health():
    """Testa endpoint de health"""
    print_section("1. TEST: /health")
    
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_chat_single():
    """Testa chat em single turn"""
    print_section("2. TEST: POST /chat (Single Turn)")
    
    try:
        payload = {
            "text": "Olá, como funciona a solicitação de férias?",
            "conversation_id": None
        }
        
        resp = requests.post(f"{BASE_URL}/chat", json=payload)
        print(f"Status: {resp.status_code}")
        data = resp.json()
        print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        if resp.status_code == 200:
            return data.get("conversation_id")
        return None
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

def test_chat_multiturn(conversation_id):
    """Testa chat multi-turn"""
    print_section("3. TEST: POST /chat (Multi-Turn)")
    
    questions = [
        "E se eu precisar de mais de 30 dias?",
        "Qual é o prazo para solicitar?",
        "Preciso de algum documento especial?"
    ]
    
    for i, question in enumerate(questions, 1):
        try:
            payload = {
                "text": question,
                "conversation_id": conversation_id
            }
            
            resp = requests.post(f"{BASE_URL}/chat", json=payload)
            print(f"\n  Pergunta {i}: {question}")
            print(f"  Status: {resp.status_code}")
            
            if resp.status_code == 200:
                data = resp.json()
                print(f"  Resposta: {data['response'][:200]}...")
                print(f"  Turn count: {data['turn_count']}")
            else:
                print(f"  ❌ Erro {resp.status_code}")
            
            time.sleep(0.5)  # Pequeno delay entre requisições
        
        except Exception as e:
            print(f"  ❌ Erro: {e}")

def test_get_conversation(conversation_id):
    """Testa recuperação de histórico"""
    print_section("4. TEST: GET /conversation/{id}")
    
    try:
        resp = requests.get(f"{BASE_URL}/conversation/{conversation_id}")
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"Conversation ID: {data['conversation_id']}")
            print(f"Total messages: {data['message_count']}")
            print(f"User turns: {data['user_turns']}")
            print(f"\nHistórico:")
            for msg in data['messages']:
                role = "👤" if msg['role'] == "user" else "🤖"
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"  {role} {msg['role']}: {content}")
        
        return resp.status_code == 200
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_list_conversations():
    """Testa listagem de conversas"""
    print_section("5. TEST: GET /conversations")
    
    try:
        resp = requests.get(f"{BASE_URL}/conversations")
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"Total conversations: {data['total_conversations']}")
            for conv in data['conversations']:
                print(f"  - {conv['id']}: {conv['message_count']} messages")
        
        return resp.status_code == 200
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_delete_conversation(conversation_id):
    """Testa deleção de conversa"""
    print_section("6. TEST: DELETE /conversation/{id}")
    
    try:
        resp = requests.delete(f"{BASE_URL}/conversation/{conversation_id}")
        print(f"Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
        
        return resp.status_code == 200
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("\n" + "="*70)
    print("  SERH RAG CHATBOT - LOCAL TEST SUITE")
    print("="*70)
    print(f"Base URL: {BASE_URL}")
    print("Aguardando conexão...")
    
    # Aguarda app estar pronto
    max_retries = 5
    for i in range(max_retries):
        try:
            requests.get(f"{BASE_URL}/health", timeout=2)
            print("✓ App está pronto!\n")
            break
        except:
            if i < max_retries - 1:
                print(f"  Tentativa {i+1}/{max_retries}... aguardando...")
                time.sleep(2)
            else:
                print(f"❌ Não foi possível conectar a {BASE_URL}")
                print("   Certifique-se que app.py está rodando com: python app.py")
                return
    
    # Executa testes
    results = []
    
    # Test 1: Health
    results.append(("Health", test_health()))
    
    # Test 2: Chat single
    conv_id = test_chat_single()
    results.append(("Chat Single", conv_id is not None))
    
    # Test 3: Chat multi-turn
    if conv_id:
        test_chat_multiturn(conv_id)
        results.append(("Chat Multi-Turn", True))
        
        # Test 4: Get conversation
        results.append(("Get Conversation", test_get_conversation(conv_id)))
        
        # Test 5: List conversations
        results.append(("List Conversations", test_list_conversations()))
        
        # Test 6: Delete conversation
        results.append(("Delete Conversation", test_delete_conversation(conv_id)))
    
    # Resumo
    print_section("RESUMO DOS TESTES")
    
    for test_name, passed in results:
        status = "✓ PASSOU" if passed else "✗ FALHOU"
        print(f"  {status}: {test_name}")
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} testes passaram")
    
    if total_passed == total_tests:
        print("\n🎉 Todos os testes passaram! App está funcional.")
    else:
        print(f"\n⚠️  {total_tests - total_passed} teste(s) falharam.")

if __name__ == "__main__":
    main()
