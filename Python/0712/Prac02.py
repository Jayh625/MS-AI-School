import os
import sys
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, \
    QLabel, QLineEdit, QPushButton, QMessageBox, QStackedWidget, QListWidget
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# 데이터베이스 설정
os.makedirs("./db", exist_ok=True)
engine = create_engine('sqlite:///db/user.db', echo=True)
Base = declarative_base()

# 사용자 모델 정의
class User(Base) :
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

    def __init__(self, username, password) :
        self.username = username
        self.password = password

# 데이터베이스 세션 설정
Session = sessionmaker(bind=engine)
session = Session()

# 회원 가입 페이지
class RegisterPage(QWidget) :
    def __init__(self, stacked_widget, main_window):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.main_window = main_window
        
        self.layout = QVBoxLayout()

        self.username_label = QLabel("Username :")
        self.username_input = QLineEdit()

        self.password_label = QLabel("Password :")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        self.register_button = QPushButton("Register")
        self.register_button.clicked.connect(self.register)

        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.register_button)

        self.setLayout(self.layout)
    
    def register(self) :
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password :
            QMessageBox.warning(self, "Error", "Please enter username ans password")
            return

        # 사용자 생성
        user = User(username, password)

        # 데이터베이스에 추가
        session.add(user)
        session.commit()

        QMessageBox.information(self, "Success", "Registration Sucessful.")
        self.stacked_widget.setCurrentIndex(1)
        
        self.main_window.show_login_page()

# 로그인 페이지
class LoginPage(QWidget) :
    def __init__(self, stacked_widget, main_window):
        super().__init__()

        self.stacked_widget = stacked_widget
        self.main_window = main_window

        self.layout = QVBoxLayout()

        self.username_label = QLabel("Username :")
        self.username_input = QLineEdit()

        self.password_label = QLabel("Password :")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.login)

        self.register_button = QPushButton("Register")
        self.register_button.clicked.connect(self.register)

        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.login_button)
        self.layout.addWidget(self.register_button)

        self.setLayout(self.layout)
    
    def login(self) :
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password :
            QMessageBox.warning(self, "Error", "Please enter username ans password")
            return

        # 사용자 조회
        user = session.query(User).filter_by(username=username, password=password).first()

        if user : 
            QMessageBox.information(self, "Success", "Login sucessful.")
            self.stacked_widget.setCurrentIndex(2)

            self.main_window.show_admin_page()
        else : 
            QMessageBox.warning(self, "Error", "Invalid username or password.")
        
    def register(self) :
        self.main_window.show_register_page()

class AdminPage(QWidget) :
    def __init__(self, main_window) :
        super().__init__()
        
        self.main_window = main_window

        self.layout = QVBoxLayout()
        
        self.user_list = QListWidget()

        self.show_user_list_button = QPushButton("Show User List")
        self.show_user_list_button.clicked.connect(self.show_user_list)

        self.logout_button = QPushButton("Logout")
        self.logout_button.clicked.connect(self.logout)

        self.layout.addWidget(self.show_user_list_button)
        self.layout.addWidget(self.user_list)
        self.layout.addWidget(self.logout_button)

        self.setLayout(self.layout)
    
    def show_user_list(self) :
        self.user_list.clear()

        # 모든 사용자 조회
        users = session.query(User).all()
        
        for user in users :
            self.user_list.addItem(user.username)
    
    def logout(self) :
        self.main_window.show_login_page()

class MainWindow(QMainWindow) :
    def __init__(self) :
        super().__init__()

        self.setWindowTitle("User Authentication")

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.register_page = RegisterPage(self.stacked_widget, self)
        self.login_page = LoginPage(self.stacked_widget, self)
        self.admin_page = AdminPage(self)

        self.stacked_widget.addWidget(self.login_page) # 초기 페이지를 로그인 페이지로 설정
        self.stacked_widget.addWidget(self.register_page)
        self.stacked_widget.addWidget(self.admin_page)

        self.show_login_page() # 초기 페이지를 보여줌
    
    def show_register_page(self) : 
        self.stacked_widget.setCurrentIndex(1) # 회원 가입 페이지의 인덱스는 1
        self.register_page.username_input.clear()
        self.register_page.password_input.clear()
    
    def show_login_page(self) :
        self.stacked_widget.setCurrentIndex(0) # 로그인 페이지의 인덱스는 0
        self.login_page.username_input.clear()
        self.login_page.password_input.clear()
    
    def show_admin_page(self) :
        self.admin_page.update_user_list()
        self.stacked_widget.setCurrentIndex(2) # 관리자 페이지의 인덱스는 2
    
    def show_register_success_message(self) :
        QMessageBox.information(self, "Success", "Registration Sucessful.") # 회원가입 성공 메시지 표시

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    Base.metadata.create_all(engine)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())