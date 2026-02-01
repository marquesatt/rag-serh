#!/usr/bin/env python3
"""Verificar assinatura da função import_files"""

from vertexai import rag
import inspect

print("=" * 60)
print("ASSINATURA DE import_files:")
print("=" * 60)

sig = inspect.signature(rag.import_files)
print(f"\nimport_files{sig}")

print("\n" + "=" * 60)
print("DOCSTRING:")
print("=" * 60)
print(rag.import_files.__doc__)

# Verificar parametros
print("\n" + "=" * 60)
print("PARÂMETROS:")
print("=" * 60)
for param_name, param in sig.parameters.items():
    print(f"\n{param_name}:")
    print(f"  Annotation: {param.annotation}")
    print(f"  Default: {param.default}")
