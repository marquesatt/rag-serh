#!/usr/bin/env python3
"""Verificar quais funções estão disponíveis no módulo vertexai.rag"""

from vertexai import rag

print("=" * 60)
print("FUNÇÕES E CLASSES DISPONÍVEIS NO MÓDULO vertexai.rag:")
print("=" * 60)

# Listar tudo que está disponível
for attr in dir(rag):
    if not attr.startswith('_'):
        obj = getattr(rag, attr)
        print(f"\n{attr}")
        if callable(obj):
            try:
                print(f"  Tipo: {type(obj).__name__}")
                if hasattr(obj, '__doc__'):
                    doc = obj.__doc__
                    if doc:
                        first_line = doc.split('\n')[0].strip()
                        print(f"  Doc: {first_line}")
            except:
                pass
