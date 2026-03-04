from t_tech.invest import Client

#TOKEN = 't.QNvaFwcLjeyyvlvkV7wXV8A3dEqaN9KlX3HH5pNnwDU_r7-WUmldmfCnNLCQnTX_WX42wqhFYPCzwDqfnh-8zQ'

TOKEN = 't.2kt-ltTDgFp83htFE9vFk5K0Gtb4Wwnf-kfGH4RkSDxfxWf9BS5Hkmp8pMIxA0gs7PHOsCiQ7IyB7BjvEpOxwg'
with Client(TOKEN) as client:
    print(client.users.get_accounts())

