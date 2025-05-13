import redis
import pickle


class RedisQueue:
    def __init__(self, name, redis_client=None, host='localhost', port=6379):
        self.key = name
        self.redis = redis_client or redis.Redis(host=host, port=port)

    def put(self, item, max_len=35):
        try:
            # Nếu hàng đợi dài hơn max_len thì pop frame cũ
            while self.redis.llen(self.key) >= max_len:
                self.redis.lpop(self.key)
            # Push frame mới vào đuôi hàng đợi
            self.redis.rpush(self.key, pickle.dumps(item))
        except Exception as e:
            print(f"[RedisQueue] Error pushing to queue: {e}")

    def get(self, block=True, timeout=1):
        try:
            if block:
                item = self.redis.blpop(self.key, timeout=timeout)
            else:
                item = self.redis.lpop(self.key)
            if item:
                _, value = item if block else (None, item)
                return pickle.loads(value)
        except Exception as e:
            print(f"[RedisQueue] Error getting from queue: {e}")
        return None
