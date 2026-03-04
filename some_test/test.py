from t_tech.invest import Client

#TOKEN = 't.QNvaFwcLjeyyvlvkV7wXV8A3dEqaN9KlX3HH5pNnwDU_r7-WUmldmfCnNLCQnTX_WX42wqhFYPCzwDqfnh-8zQ'

TOKEN = os.environ["INVEST_TOKEN"]
with Client(TOKEN) as client:
    print(client.users.get_accounts())

