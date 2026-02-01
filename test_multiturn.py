#!/usr/bin/env python3
"""Teste multi-turn com 20 perguntas mantendo contexto"""
import requests
import json

base_url = 'http://localhost:8080'
conversation_id = 'multi-turn-test-001'

perguntas = [
    # Auxílio-transporte
    "Como cadastro auxílio-transporte no SERH?",
    "Qual é o documento que preciso anexar?",
    "E se a linha não existir no sistema?",
    "Como cancelo esse benefício?",
    "Qual é o prazo para cancelamento?",
    
    # Férias
    "Como solicito férias no SERH?",
    "Qual é o saldo de férias que tenho?",
    "Posso parcelar as férias?",
    "Como acompanho o status da solicitação?",
    "E se houver erro no saldo?",
    
    # Dados consistentes/importação
    "Estou vendo dados inconsistentes. Como resolvo?",
    "A migração de dados pode causar problemas?",
    "Como valido se os dados estão corretos?",
    "O que fazer se encontrar duplicatas?",
    "Qual é o procedimento para corrigir erros?",
    
    # Frequência
    "Como lançar frequência no SERH?",
    "Posso editar lançamentos anteriores?",
    "Como recuperar faltas injustificadas?",
    "Qual é o impacto no contracheque?",
    "Existe período limite para lançamento?",
]

print("\n" + "="*80)
print("TESTE MULTI-TURN: 20 PERGUNTAS COM CONTEXTO")
print(f"Conversation ID: {conversation_id}")
print("="*80)

for i, pergunta in enumerate(perguntas, 1):
    print(f"\n[{i}/20] Pergunta: {pergunta}")
    print("-" * 80)
    
    try:
        response = requests.post(f'{base_url}/chat', json={
            'text': pergunta,
            'conversation_id': conversation_id
        }, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            resposta = result.get('response', 'Erro')
            history_len = result.get('history_length', 0)
            
            # Mostra primeira linha da resposta
            primeira_linha = resposta.split('\n')[0][:100]
            print(f"Resposta: {primeira_linha}...")
            print(f"History length: {history_len} mensagens | Turnos: {history_len//2}")
            
        else:
            print(f"Erro {response.status_code}: {response.text[:200]}")
            
    except Exception as e:
        print(f"Erro: {e}")

print("\n" + "="*80)
print("✅ TESTE CONCLUÍDO!")
print("="*80)

# Verificar histórico completo
print("\n📋 Obtendo histórico completo da conversa...")
try:
    response = requests.get(f'{base_url}/conversation/{conversation_id}', timeout=30)
    if response.status_code == 200:
        hist = response.json()
        if isinstance(hist, dict):
            print(f"\nHistórico total: {hist.get('message_count')} mensagens")
            print("\nPrimeiras 5 mensagens:")
            for i, msg in enumerate(hist.get('history', [])[:5], 1):
                if isinstance(msg, dict):
                    preview = msg.get('text', '')[:60]
                    print(f"  [{i}] ({msg.get('role', '?').upper()}): {preview}...")
        else:
            print(f"Erro ao processar histórico: formato inesperado")
    else:
        print(f"Erro ao obter histórico: {response.status_code}")
except Exception as e:
    print(f"Erro ao obter histórico: {e}")
