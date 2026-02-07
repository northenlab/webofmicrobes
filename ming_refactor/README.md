### Testing API


```
curl -X POST http://127.0.0.1:5001/api/query \
>   -H "Content-Type: application/json" \
>   -d '{"command": "microbes", "compounds": "glutamate", "level": "genus", "top": 5}'
```