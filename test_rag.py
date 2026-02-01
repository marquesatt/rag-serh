#!/usr/bin/env python3
"""Teste da API RAG localmente"""
import requests
import json
import time

base_url = 'http://localhost:8080'

# Teste 1: Pergunta sobre importação
print('\n' + '='*70)
print('TESTE 1: Problemas na importação de dados')
print('='*70)
try:
    response = requests.post(f'{base_url}/chat', json={
        'text': 'Estamos com problemas na importação de dados, dados errados e inconsistentes. Como resolver?',
        'conversation_id': 'test-001'
    }, timeout=30)
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        result = response.json()
        print(f'\nResposta:\n{result.get("response", "Erro")}')
        print(f'\nCorpus: {result.get("corpus")}')
    else:
        print(f'Erro: {response.text}')
except Exception as e:
    print(f'Erro de conexão: {e}')

time.sleep(2)

# Teste 2: Pergunta sobre auxílio transporte
print('\n' + '='*70)
print('TESTE 2: Auxílio-transporte')
print('='*70)
try:
    response = requests.post(f'{base_url}/chat', json={
        'text': 'Como cadastro auxílio-transporte no SERH?',
        'conversation_id': 'test-002'
    }, timeout=30)
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        result = response.json()
        print(f'\nResposta:\n{result.get("response", "Erro")}')
except Exception as e:
    print(f'Erro: {e}')

time.sleep(2)

# Teste 3: Pergunta sobre algo que NÃO deve estar no corpus
print('\n' + '='*70)
print('TESTE 3: Pergunta fora do escopo (presidente)')
print('='*70)
try:
    response = requests.post(f'{base_url}/chat', json={
        'text': 'Quem é o presidente do Brasil?',
        'conversation_id': 'test-003'
    }, timeout=30)
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        result = response.json()
        print(f'\nResposta:\n{result.get("response", "Erro")}')
except Exception as e:
    print(f'Erro: {e}')

time.sleep(2)

# Teste 4: Multi-turn - pergunta seguinte
print('\n' + '='*70)
print('TESTE 4: Multi-turn - Pergunta seguinte mesma conversa')
print('='*70)
try:
    response = requests.post(f'{base_url}/chat', json={
        'text': 'E como cancelo esse auxílio?',
        'conversation_id': 'test-002'
    }, timeout=30)
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        result = response.json()
        print(f'\nResposta:\n{result.get("response", "Erro")}')
        print(f'\nHistory length: {result.get("history_length")}')
except Exception as e:
    print(f'Erro: {e}')

time.sleep(2)

# Teste 5: Health check
print('\n' + '='*70)
print('TESTE 5: Health check')
print('='*70)
try:
    response = requests.get(f'{base_url}/health')
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        print('API está online!')
except Exception as e:
    print(f'Erro: {e}')
