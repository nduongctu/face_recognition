from celery import Celery

celery_app = Celery(
    'face_worker',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/1'
)

celery_app.conf.task_serializer = 'json'
celery_app.conf.result_serializer = 'json'
celery_app.conf.accept_content = ['json']
