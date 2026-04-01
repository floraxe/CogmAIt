# ✅ 1. 补全datetime相关导入：解决timezone、timedelta未定义的NameError
from datetime import datetime, timedelta, timezone

# ✅ 2. SQLAlchemy 2.0 标准导入（已修复MovedIn20Warning）
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 原有配置导入，保持不变
from app.core.config import settings


# 定义东八区区时（中国标准时间 UTC+8），代码完全不变
CST_TIMEZONE = timezone(timedelta(hours=8))

# 创建获取中国标准时间的函数，代码完全不变
def get_cn_datetime():
    """获取当前的中国标准时间(东八区，UTC+8)"""
    return datetime.now(CST_TIMEZONE)


# 创建数据库引擎，代码完全不变
engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_pre_ping=True,
    # 必要时可以添加更多连接池设置
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    # 启用回显SQL语句，便于调试
    echo=False
)

# 创建数据库会话，代码完全不变
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基本模型类，代码完全不变
Base = declarative_base()


# 创建数据库和表，代码完全不变
def init_db():
    """
    初始化数据库，如果表不存在则创建
    """
    from app.models import Model, Knowledge, KnowledgeFile, Graph, GraphNode, GraphEdge, Agent, User, Role

    # 创建所有表
    if settings.CREATE_TABLES:
        # 这里可以添加数据库初始化的代码
        # 检查数据库是否存在，如果不存在则创建（仅适用于MySQL）
        try:
            # 尝试连接到数据库
            db_uri = settings.SQLALCHEMY_DATABASE_URI
            # 从连接字符串中提取数据库名称
            db_name = settings.DB_NAME

            # 创建不指定数据库的连接URI
            server_uri = db_uri.rsplit('/', 1)[0]

            # 创建一个临时引擎来执行创建数据库的操作
            temp_engine = create_engine(server_uri)
            with temp_engine.connect() as connection:
                # 检查数据库是否存在（使用text()函数包装SQL语句）
                result = connection.execute(text(f"SHOW DATABASES LIKE '{db_name}'"))
                if not result.fetchone():
                    # 创建数据库（使用text()函数包装SQL语句）
                    connection.execute(text(f"CREATE DATABASE {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                    print(f"数据库 {db_name} 已创建")
                else:
                    print(f"数据库 {db_name} 已存在")

            # 创建所有表
            Base.metadata.create_all(bind=engine)
            print("所有表已创建或已存在")

            # 这里可以添加初始数据的创建，例如默认角色和管理员用户
            db = SessionLocal()
            try:
                # 导入用户相关函数
                from app.utils.user import create_initial_roles, create_admin_user

                # 创建初始角色
                create_initial_roles(db)
                print("初始角色已创建")

                # 创建管理员用户
                create_admin_user(db)
                print("管理员用户已创建")
            finally:
                db.close()
        except Exception as e:
            print(f"初始化数据库时出错：{str(e)}")
            raise  # 重新抛出异常，以便在应用启动时捕获
    else:
        print("自动创建表功能已禁用")