from datetime import datetime
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class ProductSchema(BaseModel):
    id: int = Field(description="Уникальный номер товара")
    name: str = Field(description="Название товара")
    price: str = Field(description="Цена товара")
    specs: str = Field(description="Краткие характеристики")


class StoreResponseSchema(BaseModel):
    store_name: str = Field(description="Название магазина")
    products: List[ProductSchema] = Field(description="Список товаров")
    support_phone: str = Field(description="Телефон службы поддержки")
    working_hours: str = Field(description="График работы")


prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        'Ты менеджер интернет-магазина. Извлеки всю информацию о товарах и контактах из текста и верни строго по структуре.\n\nКонтекст:\n{context}'
    ),
    ('human', '{text}')
])

model = ChatOllama(model='llama3.2:latest')

structured_model = model.with_structured_output(StoreResponseSchema)



def check_text(inputs_dict):
    text = inputs_dict.get("text", "")
    new_words = text.split()
    for i in new_words:
        if i in ['бомба', 'взлом', 'бомбу', 'наркотик']:
            return {"context": inputs_dict["context"], "text": "На такие вопросы не можем ответить"}
    return inputs_dict


def change_text(res_object):
    return res_object


def append_metadata(res_object):
    return {
        "store_info": res_object,
        "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "SUCCESS"
    }


chain = ( RunnableLambda(check_text)| prompt| structured_model| RunnableLambda(change_text)| RunnableLambda(append_metadata))

STORE_DATA = """
Интернет-магазин электроники "TechnoShop".

1. Товар: Ноутбук ASUS ZenBook 14
Цена: 85 000 сом.
Характеристики: 16GB RAM, SSD 512GB, Intel Core i5.

2. Товар: Смартфон Samsung Galaxy S24
Цена: 75 000 сом.
Характеристики: 8GB RAM, 256GB, камера 50 MP.

3. Товар: Беспроводные наушники AirPods Pro
Цена: 18 000 сом.
Характеристики: Активное шумоподавление, беспроводная зарядка.

Контакты: Поддержка +996 700 99 88 77, График работы: Пн-Сб с 09:00 до 20:00.
"""

inputs = [
    {
        "context": STORE_DATA,
        "text": "Выведи список товаров и контакты магазина."
    }
]

results = chain.batch(inputs)

for res in results:
    print(f" Иштетилген убакыт: {res['processed_time']}")
    print(f" Статус: {res['status']}")

    store = res['store_info']
    print(f" Дүкөн: {store.store_name}")
    print(f" Иш убактысы: {store.working_hours}")
    print(f" Байланыш: {store.support_phone}\n")

    print('Товарлар')
    for item in store.products:
        print(f"[{item.id}] {item.name} — {item.price}")
        print(f"   Мүнөздөмөсү: {item.specs}\n")