"""Гарантирует, что корень проекта доступен для импорта `app` при любом способе запуска pytest."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
