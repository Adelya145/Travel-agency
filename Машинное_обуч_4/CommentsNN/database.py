import mysql.connector
from mysql.connector import Error


class DatabaseManager:
    def __init__(self, host='localhost', user='root', password='2703', database='db_comments'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.connect()

    def connect(self):
        """Подключение к базе данных"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                print("Успешное подключение к базе данных")
        except Error as e:
            print(f"Ошибка подключения к базе данных: {e}")

    def get_all_comments(self):
        """Получить все комментарии"""
        cursor = None
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT comment_id, comment_text, comment_toxic FROM comments ORDER BY comment_id"
            cursor.execute(query)
            results = cursor.fetchall()
            return results
        except Error as e:
            print(f"Ошибка при получении комментариев: {e}")
            return []
        finally:
            if cursor:
                cursor.close()

    def save_comment(self, text, toxicity):
        """Сохранить новый комментарий"""
        cursor = None
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            cursor = self.connection.cursor()
            query = "INSERT INTO comments (comment_text, comment_toxic) VALUES (%s, %s)"
            cursor.execute(query, (text, toxicity))
            self.connection.commit()

            comment_id = cursor.lastrowid
            print(f"Комментарий сохранен с ID: {comment_id}")
            return comment_id

        except Error as e:
            print(f"Ошибка при сохранении комментария: {e}")
            self.connection.rollback()
            raise e
        finally:
            if cursor:
                cursor.close()

    def get_recent_comments(self, limit=5):
        """Получить последние комментарии"""
        cursor = None
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT comment_id, comment_text, comment_toxic FROM comments ORDER BY comment_id DESC LIMIT %s"
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            return results
        except Error as e:
            print(f"Ошибка при получении последних комментариев: {e}")
            return []
        finally:
            if cursor:
                cursor.close()

    def close(self):
        """Закрыть соединение"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Соединение с базой данных закрыто")


db_manager = DatabaseManager(
    host='localhost',
    user='root',
    password='2703',
    database='db_comments'
)


def get_db():
    """Получить экземпляр менеджера БД"""
    return db_manager