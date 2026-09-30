(function(){
  var VALID = ['landing','overview','health-score-fuzzy-decision','competitive-urgency','battle-forecast'];

  function showPage(page){
    if (VALID.indexOf(page) === -1) page = 'landing';

    document.getElementById('shell-landing').classList.toggle('active', page === 'landing');
    document.getElementById('shell-app').classList.toggle('active', page !== 'landing');

    if (page !== 'landing') {
      document.querySelectorAll('#shell-app [data-content]').forEach(function(el){
        el.style.display = (el.getAttribute('data-content') === page) ? '' : 'none';
      });
      document.querySelectorAll('#app-nav a').forEach(function(a){
        a.classList.toggle('active', a.getAttribute('data-page') === page);
      });
    }
    window.scrollTo(0,0);
  }

  function routeFromHash(){
    var hash = window.location.hash.replace('#/', '').replace('#','');
    showPage(hash || 'landing');
  }

  document.addEventListener('click', function(e){
    var link = e.target.closest('[data-page]');
    if (link) {
      // let default hash navigation happen, then sync UI
      setTimeout(routeFromHash, 0);
    }
  });

  window.addEventListener('hashchange', routeFromHash);
  routeFromHash();
})();
