"""
fase3/database.py

Gerenciador do banco de dados estruturado do hospital (SQLite).
Armazena e recupera prontuários, exames pendentes, histórico de internações
e registros de alertas da equipe médica.
"""

import sqlite3
import os
from typing import Dict, List, Any, Optional


class MedicalDatabase:
    def __init__(self, db_path: str = "data/patient_db.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_tables()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabela de Pacientes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pacientes (
                    id TEXT PRIMARY KEY,
                    nome_anonimizado TEXT NOT NULL,
                    idade INTEGER,
                    sexo TEXT,
                    leito TEXT,
                    diagnostico_admissional TEXT,
                    data_internacao TEXT
                );
            """)

            # Tabela de Prontuários / Evoluções Clínicas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prontuarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paciente_id TEXT NOT NULL,
                    data_registro TEXT NOT NULL,
                    queixa_principal TEXT,
                    sinais_vitais TEXT,
                    historico_medicamentoso TEXT,
                    alergias TEXT,
                    FOREIGN KEY(paciente_id) REFERENCES pacientes(id)
                );
            """)

            # Tabela de Exames Pendentes e Resultados
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paciente_id TEXT NOT NULL,
                    nome_exame TEXT NOT NULL,
                    status TEXT NOT NULL, -- 'PENDENTE', 'CONCLUIDO', 'CANCELADO'
                    resultado TEXT,
                    data_solicitacao TEXT NOT NULL,
                    FOREIGN KEY(paciente_id) REFERENCES pacientes(id)
                );
            """)

            conn.commit()

    def seed_sample_data(self):
        """Popula o banco com prontuários e exames de teste simplificados para o pronto-socorro."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Limpar dados antigos
            cursor.execute("DELETE FROM exames;")
            cursor.execute("DELETE FROM prontuarios;")
            cursor.execute("DELETE FROM pacientes;")

            # Paciente 1 - Dor no peito
            cursor.execute("""
                INSERT INTO pacientes (id, nome_anonimizado, idade, sexo, leito, diagnostico_admissional, data_internacao)
                VALUES ('PAC-101', 'Paciente João', 58, 'M', 'Emergência 01', 'Dor no peito a esclarecer', '2026-09-12');
            """)
            cursor.execute("""
                INSERT INTO prontuarios (paciente_id, data_registro, queixa_principal, sinais_vitais, historico_medicamentoso, alergias)
                VALUES ('PAC-101', '2026-09-12 08:30', 'Dor forte no peito queimando há 2 horas com suor frio', 
                        'Pressão: 140/90 mmHg, Batimentos: 92 bpm, Oxigênio: 96%', 'Remédio de pressão (Enalapril 20mg)', 'Dipirona (dá coceira)');
            """)
            cursor.execute("""
                INSERT INTO exames (paciente_id, nome_exame, status, resultado, data_solicitacao)
                VALUES ('PAC-101', 'Electrocardiograma (ECG)', 'CONCLUIDO', 'Alteração em parede inferior', '2026-09-12 08:40');
            """)
            cursor.execute("""
                INSERT INTO exames (paciente_id, nome_exame, status, resultado, data_solicitacao)
                VALUES ('PAC-101', 'Troponina (Exame de infarto)', 'PENDENTE', NULL, '2026-09-12 08:45');
            """)

            # Paciente 2 - Febre e Infecção
            cursor.execute("""
                INSERT INTO pacientes (id, nome_anonimizado, idade, sexo, leito, diagnostico_admissional, data_internacao)
                VALUES ('PAC-202', 'Paciente Maria', 67, 'F', 'Observação 03', 'Suspeita de Infecção / Pneumonia', '2026-09-12');
            """)
            cursor.execute("""
                INSERT INTO prontuarios (paciente_id, data_registro, queixa_principal, sinais_vitais, historico_medicamentoso, alergias)
                VALUES ('PAC-202', '2026-09-12 09:15', 'Tosse com catarro, febre de 38.8ºC e fraqueza no corpo', 
                        'Pressão: 85/55 mmHg (Baixa), Batimentos: 118 bpm (Acelerado), Oxigênio: 91%', 'Remédio de diabetes (Metformina 850mg)', 'Nenhuma');
            """)
            cursor.execute("""
                INSERT INTO exames (paciente_id, nome_exame, status, resultado, data_solicitacao)
                VALUES ('PAC-202', 'Raio-X de Tórax', 'CONCLUIDO', 'Mancha no pulmão direito', '2026-09-12 09:30');
            """)
            cursor.execute("""
                INSERT INTO exames (paciente_id, nome_exame, status, resultado, data_solicitacao)
                VALUES ('PAC-202', 'Lactato no sangue', 'PENDENTE', NULL, '2026-09-12 09:35');
            """)

            # Paciente 3 - Crise de Asma / Falta de ar (NOVO PACIENTE)
            cursor.execute("""
                INSERT INTO pacientes (id, nome_anonimizado, idade, sexo, leito, diagnostico_admissional, data_internacao)
                VALUES ('PAC-303', 'Paciente Carlos', 29, 'M', 'Nebulização / Sala Rápida', 'Crise de Asma / Falta de Ar', '2026-09-12');
            """)
            cursor.execute("""
                INSERT INTO prontuarios (paciente_id, data_registro, queixa_principal, sinais_vitais, historico_medicamentoso, alergias)
                VALUES ('PAC-303', '2026-09-12 10:00', 'Chiado no peito, falta de ar intensa e dificuldade para falar', 
                        'Pressão: 125/80 mmHg, Batimentos: 105 bpm, Respiração: 28 irpm, Oxigênio: 92%', 'Usa bombinha (Aerolin/Salbutamol) quando tem crise', 'Nenhuma');
            """)
            cursor.execute("""
                INSERT INTO exames (paciente_id, nome_exame, status, resultado, data_solicitacao)
                VALUES ('PAC-303', 'Gasometria / Exame de oxigênio no sangue', 'PENDENTE', NULL, '2026-09-12 10:05');
            """)

            conn.commit()
        print(f"[Database] Banco SQLite inicializado e populado com sucesso em: {self.db_path}")

    def get_patient_summary(self, paciente_id: str) -> Dict[str, Any]:
        """Recupera um resumo completo do paciente, prontuário e exames."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM pacientes WHERE id = ?;", (paciente_id,))
            paciente = cursor.fetchone()
            if not paciente:
                return {"erro": f"Paciente {paciente_id} não encontrado no sistema."}

            cursor.execute("SELECT * FROM prontuarios WHERE paciente_id = ? ORDER BY data_registro DESC LIMIT 1;", (paciente_id,))
            prontuario = cursor.fetchone()

            cursor.execute("SELECT * FROM exames WHERE paciente_id = ?;", (paciente_id,))
            exames = cursor.fetchall()

            return {
                "paciente": dict(paciente),
                "prontuario": dict(prontuario) if prontuario else {},
                "exames": [dict(e) for e in exames]
            }

    def get_pending_exams(self, paciente_id: str) -> List[Dict[str, Any]]:
        """Retorna apenas os exames pendentes de um paciente."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM exames WHERE paciente_id = ? AND status = 'PENDENTE';", (paciente_id,))
            return [dict(e) for e in cursor.fetchall()]


if __name__ == "__main__":
    db = MedicalDatabase()
    db.seed_sample_data()
    print("Resumo PAC-101:", db.get_patient_summary("PAC-101"))
